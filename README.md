# Rusko 3D World — Photorealistic Simulation Environment

Photorealistic 3D reconstruction of Rusko (Oulu, Finland) for drone simulation
using [Pegasus Simulator](https://pegasussimulator.github.io/PegasusSimulator/)
on NVIDIA Isaac Sim 5.1.

Two seasonal variants (summer and winter) are built from ~700 DJI drone images
each, producing textured mesh USD files with collision geometry ready for
physics-based drone simulation.

## Project Structure

```
rusko/
├── build_mesh_world.py        # Main pipeline — dense photogrammetry → USD mesh
├── build_3d_world.py          # Alt pipeline — 3DGS via 3DGRUT (Isaac Sim 6.0+)
├── 3dgrut/                    # NVIDIA 3DGRUT (submodule, provides venv + deps)
│   └── .venv/                 # Python venv with Open3D, OpenCV, OpenUSD, scipy, xatlas
├── colmap/                    # COLMAP 3.9.1 source (built with CUDA, installed to /usr/local)
├── rusko/
│   ├── rusko_summer/          # 707 JPGs, DJI FC6310R, 5472×3648, GPS-tagged
│   ├── rusko_winter/          # 724 JPGs, same camera/resolution
│   ├── output_summer/         # Pipeline output for summer
│   └── output_winter/         # Pipeline output for winter
└── README.md
```

## Pipelines

### `build_mesh_world.py` — Textured Mesh (Isaac Sim 5.1)

Produces standard textured mesh USD compatible with Isaac Sim 5.1 and Pegasus Simulator.

| Stage | Description | Est. Time |
|-------|-------------|-----------|
| 1 | COLMAP SfM — GPS-aware spatial matching, sparse reconstruction | 30-60 min |
| 2 | Dense MVS — image undistortion, patch-match stereo, fusion | 3-5 hrs |
| 3 | Mesh — Poisson reconstruction, cleaning, decimation (3M visual / 300K collision) | 20-40 min |
| 4 | Texture — multi-view projection onto 8K UV atlas | 30-60 min |
| 5 | USD Export — UsdPreviewSurface material, collision mesh with physics API | 2-5 min |

```bash
source 3dgrut/.venv/bin/activate

# Full pipeline
python build_mesh_world.py --season summer
python build_mesh_world.py --season winter

# Individual stages
python build_mesh_world.py --season summer 1      # SfM only
python build_mesh_world.py --season summer 2      # Dense MVS only
python build_mesh_world.py --season summer 3 4 5  # Mesh + Texture + Export
```

**Output:** `rusko/output_summer/export/rusko_summer.usd` + `texture_atlas.png`

### `build_3d_world.py` — 3D Gaussian Splatting (Isaac Sim 6.0+)

Uses NVIDIA 3DGRUT for neural 3D reconstruction. Produces ParticleField USDZ
which requires Isaac Sim 6.0+ (not supported by Pegasus Simulator yet).

```bash
source 3dgrut/.venv/bin/activate
python build_3d_world.py          # all stages
python build_3d_world.py 2 3      # train + export only
```

**Output:** `rusko/output/export/scene.usdz`

## Environment

| Component | Version | Notes |
|-----------|---------|-------|
| GPU | 2× RTX 3090 (24 GB each) | CUDA arch 8.6 |
| CUDA | 12.6 (driver) / 12.4 (toolkit) | |
| COLMAP | 3.9.1 with CUDA | Built from source, installed at `/usr/local/bin/colmap` |
| Python | 3.11 (in `3dgrut/.venv`) | |
| Open3D | 0.19.0 | Mesh processing |
| OpenCV | 4.13.0 | Image I/O, texture baking |
| OpenUSD (pxr) | 26.5 | USD export |
| xatlas | 0.0.11 | UV atlas generation |
| scipy | available | KDTree for camera assignment |
| Isaac Sim | 5.1 (target) | Pegasus Simulator constraint |

## Loading in Isaac Sim

1. Open Isaac Sim 5.1
2. `File > Open > rusko/output_summer/export/rusko_summer.usd`
3. Ensure `texture_atlas.png` is in the same directory as the USD file
4. Add a ground plane if needed for physics
5. Load Pegasus Simulator and spawn drone assets

## Key Configuration (build_mesh_world.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MVS_MAX_IMAGE_SIZE` | 2000 | Depth map resolution (lower = faster, less detail) |
| `POISSON_DEPTH` | 11 | Mesh resolution (9-13, higher = more detail + RAM) |
| `MESH_TARGET_VISUAL_FACES` | 3,000,000 | Visual mesh triangle count |
| `MESH_TARGET_COLLISION_FACES` | 300,000 | Physics mesh triangle count |
| `TEXTURE_ATLAS_SIZE` | 8192 | Texture resolution (4096 or 8192) |
| `MVS_GPU_INDEX` | "0,1" | GPUs for patch-match stereo |

## Data

- **Source:** DJI FC6310R drone, 5472×3648 JPEG, GPS-tagged EXIF
- **Summer:** 707 images, 5.7 GB — captured Oct 2023
- **Winter:** 724 images, 5.9 GB
- **Location:** Rusko, Oulu, Finland (~65.0°N, 25.5°E)
