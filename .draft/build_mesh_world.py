#!/usr/bin/env python3
"""
build_mesh_world.py — Dense Photogrammetry Mesh Pipeline for Isaac Sim 5.1

Creates photorealistic textured mesh USD files from drone imagery.
Unlike build_3d_world.py (which uses 3DGS/ParticleField), this produces
standard textured meshes compatible with Isaac Sim 5.1 + Pegasus Simulator.

Pipeline:
  Stage 1: COLMAP SfM — camera pose estimation (GPS-aware spatial matching)
  Stage 2: Dense MVS  — patch-match stereo + fusion → dense point cloud
  Stage 3: Mesh       — Poisson surface reconstruction + decimation
  Stage 4: Texture    — multi-view projection onto UV atlas (8K)
  Stage 5: USD Export — textured mesh + collision for Isaac Sim

Usage:
  python build_mesh_world.py --season summer         # all stages
  python build_mesh_world.py --season winter         # winter images
  python build_mesh_world.py --season summer 1 2     # stages 1-2 only
  python build_mesh_world.py --season summer 3 4 5   # mesh + texture + export

Prerequisites:
  - COLMAP with CUDA (rebuild from source for patch_match_stereo)
  - pip install xatlas
  - Already in 3dgrut venv: Open3D, OpenCV, OpenUSD (pxr), scipy
"""

import argparse
import os
import struct
import sys
import shutil
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

COLMAP_BIN = "/usr/local/bin/colmap"
COLMAP_CAMERA_MODEL = "OPENCV"
COLMAP_USE_GPU = True                       # COLMAP 3.9.1 built with CUDA
COLMAP_MAX_IMAGE_SIZE = 3200
COLMAP_USE_SPATIAL_MATCHING = True          # GPS-aware matching (much faster for drones)
COLMAP_SPATIAL_MAX_NEIGHBORS = 100

MVS_MAX_IMAGE_SIZE = 2000                   # downscale depth maps (VRAM vs quality)
MVS_GPU_INDEX = "0,1"                       # GPUs for patch-match stereo
MVS_GEOM_CONSISTENCY = True
MVS_NUM_ITERATIONS = 5

POISSON_DEPTH = 11                          # Poisson tree depth (9-13)
POISSON_TRIM = 7.0                          # trim far-from-data surfaces
MESH_TARGET_VISUAL_FACES = 3_000_000        # target triangles for rendering
MESH_TARGET_COLLISION_FACES = 300_000       # target triangles for physics

TEXTURE_ATLAS_SIZE = 8192                   # 8K atlas (4096 or 8192)
TEXTURE_BEST_K_CAMERAS = 30                # K-nearest cameras to score per face

EXPORT_UP_AXIS = "Z"
EXPORT_METERS_PER_UNIT = 1.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ============================================================================
# DERIVED PATHS (set by main() based on --season)
# ============================================================================

INPUT_IMAGES_DIR = None
OUTPUT_DIR = None
COLMAP_WS = None
DENSE_WS = None
MESH_DIR = None
TEXTURE_DIR = None
EXPORT_DIR = None


def setup_paths(season):
    global INPUT_IMAGES_DIR, OUTPUT_DIR, COLMAP_WS, DENSE_WS
    global MESH_DIR, TEXTURE_DIR, EXPORT_DIR

    INPUT_IMAGES_DIR = Path(f"rusko/rusko_{season}")
    OUTPUT_DIR = Path(f"rusko/output_{season}")
    COLMAP_WS = OUTPUT_DIR / "colmap_workspace"
    DENSE_WS = COLMAP_WS / "dense"
    MESH_DIR = OUTPUT_DIR / "mesh"
    TEXTURE_DIR = OUTPUT_DIR / "texture"
    EXPORT_DIR = OUTPUT_DIR / "export"


# ============================================================================
# HELPERS
# ============================================================================

def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


def run_cmd(cmd, cwd=None, stage="CMD"):
    log(stage, f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if result.returncode != 0:
        log(stage, f"FAILED with exit code {result.returncode}")
        sys.exit(1)


COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 5, 9: 6, 10: 8,
}


def read_colmap_cameras_binary(path):
    """Read COLMAP cameras.bin → dict of {cam_id: (model_id, width, height, params)}."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = COLMAP_CAMERA_MODEL_NUM_PARAMS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = (model_id, width, height, params)
    return cameras


def read_colmap_images_binary(path):
    """Read COLMAP images.bin → dict of {image_id: (qvec, tvec, cam_id, name)}."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # skip x, y, point3d_id per point
            images[image_id] = (qvec, tvec, cam_id, name)
    return images


def qvec2rotmat(qvec):
    """COLMAP quaternion (w,x,y,z) → 3x3 rotation matrix (world-to-camera)."""
    import numpy as np
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ])


def camera_intrinsic_matrix(model_id, params):
    """Build 3x3 intrinsic matrix K from COLMAP camera model."""
    import numpy as np
    if model_id == 0:  # SIMPLE_PINHOLE
        f, cx, cy = params[:3]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:  # OPENCV, etc. — first 4 params are fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# ============================================================================
# STAGE 1: COLMAP SfM — Camera Pose Estimation
# ============================================================================

def stage_1_colmap():
    """Run COLMAP sparse reconstruction with GPS-aware spatial matching."""
    if not INPUT_IMAGES_DIR.is_dir():
        log("SfM", f"Input images not found: {INPUT_IMAGES_DIR}")
        sys.exit(1)

    images = [f for f in INPUT_IMAGES_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if len(images) < 3:
        log("SfM", f"Need at least 3 images, found {len(images)}")
        sys.exit(1)
    log("SfM", f"Found {len(images)} images in {INPUT_IMAGES_DIR}")

    if shutil.which(COLMAP_BIN) is None:
        log("SfM", f"COLMAP not found at '{COLMAP_BIN}'. Install or rebuild with CUDA.")
        sys.exit(1)

    images_dir = COLMAP_WS / "images"
    sparse_dir = COLMAP_WS / "sparse"
    db_path = COLMAP_WS / "database.db"

    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        dest = images_dir / img.name
        if not dest.exists():
            os.symlink(img.resolve(), dest)

    log("SfM", "Step 1/3: Feature extraction")
    run_cmd([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", COLMAP_CAMERA_MODEL,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1" if COLMAP_USE_GPU else "0",
        "--SiftExtraction.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
    ], stage="SfM")

    if COLMAP_USE_SPATIAL_MATCHING:
        log("SfM", "Step 2/3: Spatial matching (GPS-aware)")
        run_cmd([
            COLMAP_BIN, "spatial_matcher",
            "--database_path", str(db_path),
            "--SpatialMatching.is_gps", "1",
            "--SpatialMatching.max_num_neighbors", str(COLMAP_SPATIAL_MAX_NEIGHBORS),
            "--SiftMatching.use_gpu", "1" if COLMAP_USE_GPU else "0",
        ], stage="SfM")
    else:
        log("SfM", "Step 2/3: Exhaustive matching")
        run_cmd([
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "1" if COLMAP_USE_GPU else "0",
        ], stage="SfM")

    log("SfM", "Step 3/3: Sparse reconstruction (mapper)")
    run_cmd([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], stage="SfM")

    model_dir = sparse_dir / "0"
    if not model_dir.is_dir():
        models = sorted(sparse_dir.iterdir())
        if models:
            model_dir = models[0]
            log("SfM", f"Using reconstruction model: {model_dir.name}")
        else:
            log("SfM", "COLMAP mapper produced no reconstruction. Check image overlap.")
            sys.exit(1)

    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        if not (model_dir / f).exists():
            log("SfM", f"Missing: {model_dir / f}")
            sys.exit(1)

    log("SfM", f"Sparse reconstruction complete: {model_dir}")


# ============================================================================
# STAGE 2: Dense MVS — Patch-Match Stereo + Fusion
# ============================================================================

def stage_2_dense_mvs():
    """Run COLMAP dense reconstruction: undistort → patch-match → fusion."""
    sparse_dir = COLMAP_WS / "sparse" / "0"
    if not sparse_dir.is_dir():
        log("MVS", "Sparse reconstruction not found. Run stage 1 first.")
        sys.exit(1)

    images_dir = COLMAP_WS / "images"
    DENSE_WS.mkdir(parents=True, exist_ok=True)

    log("MVS", "Step 1/3: Image undistortion")
    run_cmd([
        COLMAP_BIN, "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(DENSE_WS),
        "--output_type", "COLMAP",
    ], stage="MVS")

    log("MVS", "Step 2/3: Patch-match stereo (requires CUDA)")
    run_cmd([
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", str(DENSE_WS),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.max_image_size", str(MVS_MAX_IMAGE_SIZE),
        "--PatchMatchStereo.gpu_index", MVS_GPU_INDEX,
        "--PatchMatchStereo.geom_consistency", "true" if MVS_GEOM_CONSISTENCY else "false",
        "--PatchMatchStereo.num_iterations", str(MVS_NUM_ITERATIONS),
    ], stage="MVS")

    input_type = "geometric" if MVS_GEOM_CONSISTENCY else "photometric"
    depth_dir = DENSE_WS / "stereo" / "depth_maps"
    normal_dir = DENSE_WS / "stereo" / "normal_maps"
    empty_count = 0
    for dm in depth_dir.glob(f"*.{input_type}.bin"):
        if dm.stat().st_size == 0:
            dm.unlink()
            nm = normal_dir / dm.name
            if nm.exists():
                nm.unlink()
            empty_count += 1
    if empty_count:
        log("MVS", f"Removed {empty_count} empty depth map(s) to prevent fusion crash")

    fused_ply = DENSE_WS / "fused.ply"
    log("MVS", "Step 3/3: Stereo fusion → dense point cloud")
    run_cmd([
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path", str(DENSE_WS),
        "--workspace_format", "COLMAP",
        "--input_type", input_type,
        "--output_path", str(fused_ply),
        "--StereoFusion.min_num_pixels", "3",
        "--StereoFusion.max_reproj_error", "2.0",
        "--StereoFusion.use_cache", "1",
        "--StereoFusion.cache_size", "16",
    ], stage="MVS")

    if fused_ply.exists():
        size_mb = fused_ply.stat().st_size / (1024 * 1024)
        log("MVS", f"Dense point cloud: {fused_ply} ({size_mb:.1f} MB)")
    else:
        log("MVS", "Stereo fusion failed — no output PLY.")
        sys.exit(1)


# ============================================================================
# STAGE 3: Mesh Reconstruction — Poisson + Decimation
# ============================================================================

def stage_3_mesh():
    """Poisson surface reconstruction from dense point cloud."""
    import numpy as np
    import open3d as o3d

    fused_ply = DENSE_WS / "fused.ply"
    if not fused_ply.exists():
        log("MESH", "Dense point cloud not found. Run stage 2 first.")
        sys.exit(1)

    MESH_DIR.mkdir(parents=True, exist_ok=True)

    log("MESH", f"Loading dense point cloud: {fused_ply}")
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    log("MESH", f"Loaded {len(pcd.points)} points")

    if len(pcd.normals) == 0:
        log("MESH", "Estimating normals")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    log("MESH", f"Poisson reconstruction (depth={POISSON_DEPTH})")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH
    )
    log("MESH", f"Raw mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    densities_np = np.asarray(densities)
    density_threshold = np.quantile(densities_np, 0.01)
    vertices_to_remove = densities_np < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    log("MESH", f"After density trim: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangle_clusters = np.asarray(triangle_clusters)
    if len(cluster_n_triangles) > 1:
        largest_cluster = np.argmax(cluster_n_triangles)
        triangles_to_remove = triangle_clusters != largest_cluster
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        log("MESH", f"Kept largest component: {len(mesh.triangles)} faces")

    mesh.compute_vertex_normals()

    poisson_path = MESH_DIR / "poisson_mesh.ply"
    o3d.io.write_triangle_mesh(str(poisson_path), mesh)
    log("MESH", f"Saved full mesh: {poisson_path}")

    num_faces = len(mesh.triangles)

    visual_target = min(num_faces, MESH_TARGET_VISUAL_FACES)
    if num_faces > visual_target:
        visual_mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=visual_target
        )
        visual_mesh.compute_vertex_normals()
    else:
        visual_mesh = mesh
    visual_path = MESH_DIR / "visual_mesh.ply"
    o3d.io.write_triangle_mesh(str(visual_path), visual_mesh)
    log("MESH", f"Visual mesh: {len(visual_mesh.triangles)} faces → {visual_path}")

    collision_target = min(num_faces, MESH_TARGET_COLLISION_FACES)
    if num_faces > collision_target:
        collision_mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=collision_target
        )
    else:
        collision_mesh = mesh
    collision_path = MESH_DIR / "collision_mesh.ply"
    o3d.io.write_triangle_mesh(str(collision_path), collision_mesh)
    log("MESH", f"Collision mesh: {len(collision_mesh.triangles)} faces → {collision_path}")


# ============================================================================
# STAGE 4: Texture Mapping — Multi-View Projection onto UV Atlas
# ============================================================================

def stage_4_texture():
    """Generate UV atlas and bake textures from original camera images (GPU-accelerated)."""
    import numpy as np
    import cv2
    import open3d as o3d
    import torch
    import torch.nn.functional as F
    import time

    visual_path = MESH_DIR / "visual_mesh.ply"
    if not visual_path.exists():
        log("TEX", "Visual mesh not found. Run stage 3 first.")
        sys.exit(1)

    dense_cam_path = DENSE_WS / "sparse" / "cameras.bin"
    dense_img_path = DENSE_WS / "sparse" / "images.bin"
    dense_images_dir = DENSE_WS / "images"
    if not dense_cam_path.exists():
        log("TEX", "Undistorted cameras not found. Run stage 2 first.")
        sys.exit(1)

    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # --- Load mesh ---
    log("TEX", "Loading visual mesh")
    mesh = o3d.io.read_triangle_mesh(str(visual_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)
    log("TEX", f"Mesh: {len(vertices)} verts, {len(triangles)} faces")

    # --- Load cameras ---
    log("TEX", "Loading undistorted camera parameters")
    colmap_cameras = read_colmap_cameras_binary(str(dense_cam_path))
    colmap_images = read_colmap_images_binary(str(dense_img_path))

    cam_list = []
    for img_id in sorted(colmap_images.keys()):
        qvec, tvec, cam_id, name = colmap_images[img_id]
        model_id, width, height, params = colmap_cameras[cam_id]
        R = qvec2rotmat(qvec)
        t = np.array(tvec)
        K = camera_intrinsic_matrix(model_id, params)
        center = -R.T @ t
        img_path = dense_images_dir / name
        if img_path.exists():
            cam_list.append({
                "R": R, "t": t, "K": K, "center": center,
                "width": int(width), "height": int(height),
                "image_path": str(img_path),
            })
    log("TEX", f"Loaded {len(cam_list)} cameras")
    if len(cam_list) == 0:
        log("TEX", "No valid cameras found.")
        sys.exit(1)

    # --- Generate UV atlas (GPU cube-map projection, replaces xatlas) ---
    log("TEX", f"Generating UV atlas ({TEXTURE_ATLAS_SIZE}x{TEXTURE_ATLAS_SIZE}) via cube-map projection (GPU)")
    t0 = time.time()

    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    tris_t = torch.tensor(triangles, dtype=torch.int64, device=device)
    num_faces = len(tris_t)

    v0_t = verts_t[tris_t[:, 0]]
    v1_t = verts_t[tris_t[:, 1]]
    v2_t = verts_t[tris_t[:, 2]]
    face_normals_uv = F.normalize(torch.cross(v1_t - v0_t, v2_t - v0_t, dim=1), dim=1)

    directions = torch.tensor([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
    ], dtype=torch.float32, device=device)
    proj_axes = [(1,2), (1,2), (0,2), (0,2), (0,1), (0,1)]

    face_clusters = torch.argmax(face_normals_uv @ directions.t(), dim=1)

    cluster_areas = []
    for ci in range(6):
        mask = (face_clusters == ci)
        if mask.sum() == 0:
            cluster_areas.append(0.0)
            continue
        ax1, ax2 = proj_axes[ci]
        ct = tris_t[mask]
        p0 = verts_t[ct[:, 0]][:, [ax1, ax2]]
        p1 = verts_t[ct[:, 1]][:, [ax1, ax2]]
        p2 = verts_t[ct[:, 2]][:, [ax1, ax2]]
        e1 = p1 - p0; e2 = p2 - p0
        cluster_areas.append(
            (0.5 * torch.abs(e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0])).sum().item()
        )

    total_area = max(sum(cluster_areas), 1e-12)
    padding = 4
    atlas_sz = TEXTURE_ATLAS_SIZE

    strip_heights = [max(1, int(a / total_area * atlas_sz)) for a in cluster_areas]
    diff = atlas_sz - sum(strip_heights)
    largest_ci = max(range(6), key=lambda i: cluster_areas[i])
    strip_heights[largest_ci] += diff

    y_offsets = []
    cum_y = 0
    for h in strip_heights:
        y_offsets.append(cum_y)
        cum_y += h

    face_uvs = torch.zeros((num_faces, 3, 2), dtype=torch.float32, device=device)

    for ci in range(6):
        mask = (face_clusters == ci)
        if mask.sum() == 0:
            continue
        ax1, ax2 = proj_axes[ci]
        ct = tris_t[mask]
        p0 = verts_t[ct[:, 0]][:, [ax1, ax2]]
        p1 = verts_t[ct[:, 1]][:, [ax1, ax2]]
        p2 = verts_t[ct[:, 2]][:, [ax1, ax2]]

        all_pts = torch.cat([p0, p1, p2], dim=0)
        pmin = all_pts.min(dim=0).values
        pmax = all_pts.max(dim=0).values
        extent = (pmax - pmin).clamp(min=1e-6)

        usable_w = float(atlas_sz - 2 * padding)
        usable_h = float(strip_heights[ci] - 2 * padding)
        if usable_w <= 0 or usable_h <= 0:
            continue

        scale = min(usable_w / extent[0].item(), usable_h / extent[1].item())
        final_w = extent[0].item() * scale
        final_h = extent[1].item() * scale
        x_off = padding + (usable_w - final_w) / 2.0
        y_off = y_offsets[ci] + padding + (usable_h - final_h) / 2.0
        offset = torch.tensor([x_off, y_off], dtype=torch.float32, device=device)

        face_uvs[mask, 0, :] = (p0 - pmin) * scale + offset
        face_uvs[mask, 1, :] = (p1 - pmin) * scale + offset
        face_uvs[mask, 2, :] = (p2 - pmin) * scale + offset

    vmapping = triangles.ravel().astype(np.uint32)
    uvs = face_uvs.cpu().numpy().reshape(-1, 2).astype(np.float32)
    new_indices = np.arange(3 * num_faces, dtype=np.uint32).reshape(num_faces, 3)

    del face_uvs, face_normals_uv, v0_t, v1_t, v2_t, verts_t, tris_t
    torch.cuda.empty_cache()

    log("TEX", f"UV atlas done in {time.time()-t0:.1f}s: {len(uvs)} UV verts, {len(new_indices)} faces")
    for ci in range(6):
        n = (face_clusters.cpu() == ci).sum().item()
        d = ["+X","-X","+Y","-Y","+Z","-Z"][ci]
        if n > 0:
            log("TEX", f"  {d}: {n} faces, {strip_heights[ci]}px strip")

    new_vertices_3d = vertices[vmapping]

    # --- GPU: Compute per-face centers and normals ---
    log("TEX", "Computing face properties (GPU)")
    v0_t = torch.tensor(new_vertices_3d[new_indices[:, 0]], dtype=torch.float32, device=device)
    v1_t = torch.tensor(new_vertices_3d[new_indices[:, 1]], dtype=torch.float32, device=device)
    v2_t = torch.tensor(new_vertices_3d[new_indices[:, 2]], dtype=torch.float32, device=device)

    face_centers_t = (v0_t + v1_t + v2_t) / 3.0
    face_normals_t = torch.cross(v1_t - v0_t, v2_t - v0_t, dim=1)
    face_normals_t = F.normalize(face_normals_t, dim=1)

    # --- GPU: Assign best camera per face ---
    log("TEX", "Assigning best camera to each face (GPU)")
    cam_positions_t = torch.tensor(
        np.array([c["center"] for c in cam_list]), dtype=torch.float32, device=device
    )
    num_faces = len(new_indices)
    num_cams = len(cam_list)
    k = min(TEXTURE_BEST_K_CAMERAS, num_cams)

    best_cam_per_face = np.zeros(num_faces, dtype=np.int32)
    chunk_size = 200_000

    for start in range(0, num_faces, chunk_size):
        end = min(start + chunk_size, num_faces)
        centers = face_centers_t[start:end]
        normals = face_normals_t[start:end]

        dists = torch.cdist(centers, cam_positions_t)
        _, topk_idx = torch.topk(dists, k, dim=1, largest=False)

        topk_pos = cam_positions_t[topk_idx]
        view_dirs = topk_pos - centers.unsqueeze(1)
        view_dists = torch.norm(view_dirs, dim=2, keepdim=True)
        view_dirs_n = view_dirs / (view_dists + 1e-8)

        cos_a = (normals.unsqueeze(1) * view_dirs_n).sum(dim=2)
        scores = torch.where(
            cos_a > 0.1,
            cos_a / (view_dists.squeeze(-1) + 1e-8),
            torch.tensor(-float('inf'), device=device),
        )
        best_k = torch.argmax(scores, dim=1)
        best_cam_per_face[start:end] = topk_idx[
            torch.arange(end - start, device=device), best_k
        ].cpu().numpy()

        if start % 500_000 == 0 and start > 0:
            log("TEX", f"  scored {start}/{num_faces} faces")

    del v0_t, v1_t, v2_t, face_centers_t, face_normals_t, cam_positions_t
    torch.cuda.empty_cache()
    log("TEX", "Camera assignment complete")

    # --- Rasterize UV triangles into face_id_map ---
    log("TEX", "Rasterizing UV triangles into atlas")
    atlas_h = TEXTURE_ATLAS_SIZE
    atlas_w = TEXTURE_ATLAS_SIZE
    face_id_map = np.full((atlas_h, atlas_w), -1, dtype=np.int32)

    uv_pixel = uvs.copy()
    uv_pixel[:, 0] = np.clip(uv_pixel[:, 0], 0, atlas_w - 1)
    uv_pixel[:, 1] = np.clip(uv_pixel[:, 1], 0, atlas_h - 1)

    uv_tri_pixels = uv_pixel[new_indices].astype(np.int32)

    for i in range(num_faces):
        cv2.fillConvexPoly(face_id_map, uv_tri_pixels[i], int(i))
        if i % 500_000 == 0 and i > 0:
            log("TEX", f"  rasterized {i}/{num_faces} faces")

    valid_mask = face_id_map >= 0
    valid_y, valid_x = np.where(valid_mask)
    valid_face_ids = face_id_map[valid_y, valid_x]
    num_valid = len(valid_y)
    log("TEX", f"Atlas coverage: {num_valid} texels ({100*num_valid/(atlas_h*atlas_w):.1f}%)")

    # --- Compute barycentric coordinates for each valid texel ---
    log("TEX", "Computing barycentric coordinates")
    P = np.stack([valid_x.astype(np.float64) + 0.5,
                  valid_y.astype(np.float64) + 0.5], axis=1)

    tri_a = uv_pixel[new_indices[valid_face_ids, 0]].astype(np.float64)
    tri_b = uv_pixel[new_indices[valid_face_ids, 1]].astype(np.float64)
    tri_c = uv_pixel[new_indices[valid_face_ids, 2]].astype(np.float64)

    e0 = tri_c - tri_a
    e1 = tri_b - tri_a
    e2 = P - tri_a

    dot00 = (e0 * e0).sum(axis=1)
    dot01 = (e0 * e1).sum(axis=1)
    dot02 = (e0 * e2).sum(axis=1)
    dot11 = (e1 * e1).sum(axis=1)
    dot12 = (e1 * e2).sum(axis=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)
    bary_u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    bary_v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    bary_w = 1.0 - bary_u - bary_v

    # --- Compute 3D positions for each texel ---
    pos_a = new_vertices_3d[new_indices[valid_face_ids, 0]]
    pos_b = new_vertices_3d[new_indices[valid_face_ids, 1]]
    pos_c = new_vertices_3d[new_indices[valid_face_ids, 2]]
    pos_3d = bary_w[:, None] * pos_a + bary_v[:, None] * pos_b + bary_u[:, None] * pos_c

    texel_cam_idx = best_cam_per_face[valid_face_ids]

    # --- GPU: Bake textures from camera images ---
    log("TEX", "Baking textures from camera images (GPU)")
    atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)

    R_all = torch.tensor(
        np.stack([c["R"] for c in cam_list]), dtype=torch.float32, device=device
    )
    t_all = torch.tensor(
        np.stack([c["t"] for c in cam_list]), dtype=torch.float32, device=device
    )
    K_all = torch.tensor(
        np.stack([c["K"] for c in cam_list]), dtype=torch.float32, device=device
    )
    pos_3d_t = torch.tensor(pos_3d, dtype=torch.float32, device=device)
    texel_cam_idx_t = torch.tensor(texel_cam_idx, dtype=torch.long, device=device)

    for cam_idx in range(num_cams):
        cam_mask = texel_cam_idx_t == cam_idx
        n_texels = cam_mask.sum().item()
        if n_texels == 0:
            continue

        cam = cam_list[cam_idx]
        img = cv2.imread(cam["image_path"])
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        img_t = torch.tensor(
            img, dtype=torch.float32, device=device
        ).permute(2, 0, 1).unsqueeze(0)

        R = R_all[cam_idx]
        t_vec = t_all[cam_idx]
        K = K_all[cam_idx]

        pts = pos_3d_t[cam_mask]
        p_cam = pts @ R.t() + t_vec.unsqueeze(0)

        z = p_cam[:, 2].clamp(min=1e-8)
        px = K[0, 0] * p_cam[:, 0] / z + K[0, 2]
        py = K[1, 1] * p_cam[:, 1] / z + K[1, 2]

        grid_x = 2.0 * px / (img_w - 1) - 1.0
        grid_y = 2.0 * py / (img_h - 1) - 1.0

        valid = (p_cam[:, 2] > 0.1) & (grid_x.abs() <= 1) & (grid_y.abs() <= 1)

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            img_t, grid, mode='bilinear', align_corners=True, padding_mode='zeros'
        )
        colors = sampled[0, :, 0, :].t()

        cam_mask_indices = torch.where(cam_mask)[0]
        valid_indices = cam_mask_indices[valid].cpu().numpy()
        valid_colors = colors[valid].cpu().numpy()

        ay = valid_y[valid_indices]
        ax = valid_x[valid_indices]
        atlas[ay, ax] = np.clip(valid_colors, 0, 255).astype(np.uint8)

        if cam_idx % 50 == 0:
            log("TEX", f"  camera {cam_idx}/{num_cams}: {n_texels} texels, {valid.sum().item()} sampled")

    del pos_3d_t, texel_cam_idx_t, R_all, t_all, K_all
    torch.cuda.empty_cache()

    # --- Fill gaps ---
    log("TEX", "Filling atlas gaps (inpainting)")
    gap_mask = (~valid_mask).astype(np.uint8)
    filled_mask = (atlas.sum(axis=2) > 0).astype(np.uint8)
    inpaint_mask = np.maximum(gap_mask, 1 - filled_mask)
    atlas = cv2.inpaint(atlas, inpaint_mask, 3, cv2.INPAINT_TELEA)

    atlas_path = TEXTURE_DIR / "texture_atlas.png"
    cv2.imwrite(str(atlas_path), atlas)
    log("TEX", f"Saved texture atlas: {atlas_path}")

    # --- Save UV data for Stage 5 ---
    uv_data_path = TEXTURE_DIR / "uv_data.npz"
    uv_normalized = uvs.copy().astype(np.float32)
    uv_normalized[:, 0] /= atlas_w
    uv_normalized[:, 1] = 1.0 - (uv_normalized[:, 1] / atlas_h)
    np.savez_compressed(
        str(uv_data_path),
        vmapping=vmapping,
        uv_normalized=uv_normalized,
        new_indices=new_indices,
    )
    log("TEX", f"Saved UV data: {uv_data_path}")


# ============================================================================
# STAGE 5: USD Export — Textured Mesh + Collision for Isaac Sim
# ============================================================================

def stage_5_usd_export():
    """Export textured visual mesh + collision mesh as USD for Isaac Sim 5.1."""
    import numpy as np
    import open3d as o3d
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    visual_path = MESH_DIR / "visual_mesh.ply"
    collision_path = MESH_DIR / "collision_mesh.ply"
    atlas_path = TEXTURE_DIR / "texture_atlas.png"
    uv_data_path = TEXTURE_DIR / "uv_data.npz"

    for p, stage_needed in [(visual_path, "3"), (atlas_path, "4"), (uv_data_path, "4")]:
        if not p.exists():
            log("USD", f"Missing {p}. Run stage {stage_needed} first.")
            sys.exit(1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log("USD", "Loading mesh and UV data")
    visual_mesh = o3d.io.read_triangle_mesh(str(visual_path))
    vertices = np.asarray(visual_mesh.vertices, dtype=np.float32)
    triangles = np.asarray(visual_mesh.triangles, dtype=np.int32)

    uv_data = np.load(str(uv_data_path))
    vmapping = uv_data["vmapping"]
    uv_normalized = uv_data["uv_normalized"]
    new_indices = uv_data["new_indices"]

    has_collision = collision_path.exists()
    if has_collision:
        coll_mesh = o3d.io.read_triangle_mesh(str(collision_path))
        coll_verts = np.asarray(coll_mesh.vertices, dtype=np.float32)
        coll_tris = np.asarray(coll_mesh.triangles, dtype=np.int32)

    # --- Copy texture to export dir ---
    season = INPUT_IMAGES_DIR.name.replace("rusko_", "")
    export_texture = EXPORT_DIR / "texture_atlas.png"
    shutil.copy2(str(atlas_path), str(export_texture))

    output_usd = EXPORT_DIR / f"rusko_{season}.usd"

    # --- Create USD stage ---
    log("USD", f"Creating USD: {output_usd}")
    stage = Usd.Stage.CreateNew(str(output_usd))
    up_axis = UsdGeom.Tokens.z if EXPORT_UP_AXIS == "Z" else UsdGeom.Tokens.y
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, EXPORT_METERS_PER_UNIT)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # --- Visual mesh ---
    log("USD", "Writing visual mesh")
    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Terrain")
    mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh_prim.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh_prim.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(triangles.ravel())
    )
    mesh_prim.CreateSubdivisionSchemeAttr("none")

    # UV primvar (faceVarying: one UV per face-vertex)
    uv_per_face_vertex = uv_normalized[new_indices.ravel()]
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    uv_primvar = primvars_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    uv_primvar.Set(Vt.Vec2fArray.FromNumpy(uv_per_face_vertex))

    # --- Material with UsdPreviewSurface ---
    log("USD", "Creating textured material")
    material = UsdShade.Material.Define(stage, "/World/Looks/TerrainMaterial")

    shader = UsdShade.Shader.Define(stage, "/World/Looks/TerrainMaterial/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    tex_reader = UsdShade.Shader.Define(stage, "/World/Looks/TerrainMaterial/DiffuseTexture")
    tex_reader.CreateIdAttr("UsdUVTexture")
    tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture_atlas.png")
    )
    tex_reader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    st_reader = UsdShade.Shader.Define(stage, "/World/Looks/TerrainMaterial/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.GetOutput("result")
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_reader.GetOutput("rgb")
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))

    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    # --- Collision mesh ---
    if has_collision:
        log("USD", "Writing collision mesh")
        coll_prim = UsdGeom.Mesh.Define(stage, "/World/CollisionMesh")
        coll_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(coll_verts))
        coll_prim.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(coll_tris), 3, dtype=np.int32))
        )
        coll_prim.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(coll_tris.ravel())
        )
        coll_prim.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Imageable(coll_prim).MakeInvisible()

        UsdPhysics.CollisionAPI.Apply(coll_prim.GetPrim())
        coll_prim.GetPrim().GetAttribute("physics:collisionEnabled").Set(True)
        mesh_coll_api = UsdPhysics.MeshCollisionAPI.Apply(coll_prim.GetPrim())
        mesh_coll_api.CreateApproximationAttr().Set("none")

    # --- Save ---
    stage.GetRootLayer().Save()
    size_mb = output_usd.stat().st_size / (1024 * 1024)
    log("USD", f"Done! {output_usd} ({size_mb:.1f} MB)")
    log("USD", f"Load in Isaac Sim: File > Open > {output_usd.resolve()}")


# ============================================================================
# MAIN
# ============================================================================

STAGES = {
    "1": ("COLMAP SfM — Camera Pose Estimation", stage_1_colmap),
    "2": ("Dense MVS — Patch-Match Stereo + Fusion", stage_2_dense_mvs),
    "3": ("Mesh — Poisson Reconstruction + Decimation", stage_3_mesh),
    "4": ("Texture — Multi-View Projection Atlas", stage_4_texture),
    "5": ("USD Export — Isaac Sim 5.1", stage_5_usd_export),
}


def main():
    parser = argparse.ArgumentParser(
        description="Dense photogrammetry mesh pipeline for Isaac Sim 5.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--season", required=True, choices=["summer", "winter"],
        help="Which image set to process",
    )
    parser.add_argument(
        "stages", nargs="*", default=list(STAGES.keys()),
        help="Stage numbers to run (default: all). E.g. '1 2' or '3 4 5'",
    )
    args = parser.parse_args()

    for s in args.stages:
        if s not in STAGES:
            parser.error(f"Unknown stage: {s}. Valid: {', '.join(STAGES.keys())}")

    setup_paths(args.season)

    if not INPUT_IMAGES_DIR.is_dir():
        log("MAIN", f"Image directory not found: {INPUT_IMAGES_DIR}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Dense Photogrammetry → Isaac Sim 5.1 USD")
    print(f"  Season: {args.season}")
    print(f"  Stages: {', '.join(args.stages)}")
    print("=" * 60)

    for s in args.stages:
        name, func = STAGES[s]
        print(f"\n{'=' * 60}")
        print(f"  Stage {s}: {name}")
        print(f"{'=' * 60}")
        func()

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print("=" * 60)
    if "5" in args.stages:
        season = args.season
        output_file = EXPORT_DIR / f"rusko_{season}.usd"
        print(f"  Output:  {output_file}")
        print(f"  Texture: {EXPORT_DIR / 'texture_atlas.png'}")
        print(f"  Load in Isaac Sim: File > Open > {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
