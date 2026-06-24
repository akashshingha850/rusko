#!/usr/bin/env python3
"""
Images → Isaac Sim 3D World via NVIDIA 3DGRUT

Pipeline (run individually or all at once):
  python build_3d_world.py 1       # COLMAP: camera pose estimation
  python build_3d_world.py 2       # 3DGRUT: train 3D Gaussian model
  python build_3d_world.py 3       # Export: USD for Isaac Sim
  python build_3d_world.py         # run all stages
  python build_3d_world.py 2 3     # run stages 2 and 3 only

Prerequisites:
  - COLMAP installed: sudo apt install colmap  (or build from source)
  - 3DGRUT cloned and installed: https://github.com/nv-tlabs/3dgrut
      git clone --recursive https://github.com/nv-tlabs/3dgrut.git
      cd 3dgrut && ./install_env_uv.sh && source .venv/bin/activate
  - CUDA toolkit (11.8, 12.4, 12.6, or 12.8)

Usage:
  1. Edit the configuration variables below
  2. python build_3d_world.py [stage_numbers...]
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION — edit these to match your setup
# ============================================================================

# === PATHS ===
INPUT_IMAGES_DIR = "rusko/rusko_summer"  # folder with your scene photos
OUTPUT_DIR = "rusko/output"              # all outputs go here
THREEDGRUT_DIR = "3dgrut"                # cloned 3DGRUT repo root

# === COLMAP SETTINGS ===
COLMAP_BIN = "colmap"                            # path to colmap binary
COLMAP_CAMERA_MODEL = "OPENCV"                   # SIMPLE_PINHOLE, PINHOLE, OPENCV, SIMPLE_RADIAL
COLMAP_USE_GPU = True                            # set False if no GPU or COLMAP built without CUDA
COLMAP_MAX_IMAGE_SIZE = 3200                     # max image dimension for feature extraction

# === 3DGRUT TRAINING ===
CONFIG_NAME = "apps/colmap_3dgut.yaml"           # 3dgut = rasterization (good for RTX 30-series)
                                                 # 3dgrt = ray tracing (needs strong RT cores)
N_ITERATIONS = 30_000                            # training steps (more = better quality, slower)
DOWNSAMPLE_FACTOR = 1                            # 2 or 4 to reduce VRAM usage on large images
NUM_WORKERS = 8                                  # dataloader workers
MAX_SH_DEGREE = 3                                # spherical harmonics degree (0-3, higher = better view-dependent effects)
OPTIMIZER_TYPE = "adam"                           # "adam" or "selective_adam"
POSITION_LR = 0.00016                            # position learning rate
DENSITY_LR = 0.05                                # density (opacity) learning rate
SCALE_LR = 0.005                                 # scale learning rate
LAMBDA_L1 = 0.8                                  # L1 loss weight
LAMBDA_SSIM = 0.2                                # SSIM loss weight
WITH_GUI = False                                 # True to show interactive training viewer

# === GPU ===
TORCH_CUDA_ARCH_LIST = "8.6"                     # RTX 3090 = 8.6, RTX 4090 = 8.9, RTX 5090 = 10.0+

# === EXPORT ===
EXPORT_FORMAT = "standard"                       # "standard" (ParticleField, recommended) or "nurec"
EXPORT_HALF_PRECISION = True                     # float16 for smaller files
EXPORT_UP_AXIS = "z"                             # "z" for Isaac Sim (Z-up), "y" for other apps

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

COLMAP_WS = Path(OUTPUT_DIR) / "colmap_workspace"
TRAINING_DIR = Path(OUTPUT_DIR) / "training"
EXPORT_DIR = Path(OUTPUT_DIR) / "export"
EXPERIMENT_NAME = "scene"


def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


def run_cmd(cmd, cwd=None, stage="CMD", env_extra=None):
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST
    if env_extra:
        env.update(env_extra)
    log(stage, f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        log(stage, f"FAILED with exit code {result.returncode}")
        sys.exit(1)


# ============================================================================
# STAGE 1: COLMAP — Camera Pose Estimation
# ============================================================================
def stage_1_colmap():
    """Run COLMAP sparse reconstruction on input images."""
    if not Path(INPUT_IMAGES_DIR).is_dir():
        log("COLMAP", f"Input images directory not found: {INPUT_IMAGES_DIR}")
        sys.exit(1)

    images = [
        f for f in Path(INPUT_IMAGES_DIR).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(images) < 3:
        log("COLMAP", f"Need at least 3 images, found {len(images)}")
        sys.exit(1)
    log("COLMAP", f"Found {len(images)} images in {INPUT_IMAGES_DIR}")

    if shutil.which(COLMAP_BIN) is None:
        log("COLMAP", f"COLMAP not found at '{COLMAP_BIN}'. Install: sudo apt install colmap")
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

    log("COLMAP", "Step 1/3: Feature extraction")
    run_cmd([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", COLMAP_CAMERA_MODEL,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1" if COLMAP_USE_GPU else "0",
        "--SiftExtraction.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
    ], stage="COLMAP")

    log("COLMAP", "Step 2/3: Exhaustive matching")
    run_cmd([
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1" if COLMAP_USE_GPU else "0",
    ], stage="COLMAP")

    log("COLMAP", "Step 3/3: Sparse reconstruction (mapper)")
    run_cmd([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], stage="COLMAP")

    model_dir = sparse_dir / "0"
    if not model_dir.is_dir():
        models = sorted(sparse_dir.iterdir())
        if models:
            model_dir = models[0]
            log("COLMAP", f"Using reconstruction model: {model_dir.name}")
        else:
            log("COLMAP", "COLMAP mapper produced no reconstruction. Check image overlap.")
            sys.exit(1)

    required = ["cameras.bin", "images.bin", "points3D.bin"]
    for f in required:
        if not (model_dir / f).exists():
            log("COLMAP", f"Missing expected file: {model_dir / f}")
            sys.exit(1)

    log("COLMAP", f"Sparse reconstruction complete: {model_dir}")


# ============================================================================
# STAGE 2: 3DGRUT — 3D Gaussian Training
# ============================================================================
def stage_2_train():
    """Train 3DGRUT model on the COLMAP output."""
    if not (COLMAP_WS / "sparse" / "0").is_dir():
        log("TRAIN", f"COLMAP output not found. Run stage 1 first.")
        sys.exit(1)

    threedgrut_path = Path(THREEDGRUT_DIR)
    if not (threedgrut_path / "train.py").is_file():
        log("TRAIN", f"train.py not found in 3DGRUT repo: {THREEDGRUT_DIR}")
        sys.exit(1)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    colmap_ws_abs = str(COLMAP_WS.resolve())
    training_dir_abs = str(TRAINING_DIR.resolve())

    cmd = [
        sys.executable, "train.py",
        "--config-name", CONFIG_NAME,
        f"path={colmap_ws_abs}",
        f"out_dir={training_dir_abs}",
        f"experiment_name={EXPERIMENT_NAME}",
        f"n_iterations={N_ITERATIONS}",
        f"num_workers={NUM_WORKERS}",
        f"dataset.downsample_factor={DOWNSAMPLE_FACTOR}",
        f"optimizer.type={OPTIMIZER_TYPE}",
        f"optimizer.params.positions.lr={POSITION_LR}",
        f"optimizer.params.density.lr={DENSITY_LR}",
        f"optimizer.params.scale.lr={SCALE_LR}",
        f"loss.lambda_l1={LAMBDA_L1}",
        f"loss.lambda_ssim={LAMBDA_SSIM}",
        f"model.progressive_training.max_n_features={MAX_SH_DEGREE}",
        f"with_gui={WITH_GUI}",
        "export_usd.enabled=true",
    ]

    log("TRAIN", f"Starting 3DGRUT training ({N_ITERATIONS} iterations)")
    run_cmd(cmd, cwd=THREEDGRUT_DIR, stage="TRAIN")

    ckpt_path = TRAINING_DIR / EXPERIMENT_NAME / "ckpt_last.pt"
    if not ckpt_path.exists():
        ckpt_candidates = sorted(
            glob.glob(str(TRAINING_DIR / EXPERIMENT_NAME / "ckpt_*.pt"))
        )
        if ckpt_candidates:
            ckpt_path = Path(ckpt_candidates[-1])
        else:
            log("TRAIN", f"No checkpoint found in {TRAINING_DIR / EXPERIMENT_NAME}")
            sys.exit(1)

    log("TRAIN", f"Training complete. Checkpoint: {ckpt_path}")


# ============================================================================
# STAGE 3: Export — USD for Isaac Sim
# ============================================================================
def stage_3_export():
    """Export trained model to USD for Isaac Sim."""
    ckpt_path = TRAINING_DIR / EXPERIMENT_NAME / "ckpt_last.pt"
    if not ckpt_path.exists():
        ckpt_candidates = sorted(
            glob.glob(str(TRAINING_DIR / EXPERIMENT_NAME / "ckpt_*.pt"))
        )
        if ckpt_candidates:
            ckpt_path = Path(ckpt_candidates[-1])
        else:
            log("EXPORT", f"No checkpoint found. Run stage 2 first.")
            sys.exit(1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = EXPORT_DIR / "scene.usdz"

    cmd = [
        sys.executable, "-m", "threedgrut.export.scripts.export_usd",
        "--checkpoint", str(ckpt_path.resolve()),
        "--output", str(output_file.resolve()),
        "--dataset", str(COLMAP_WS.resolve()),
        "--format", EXPORT_FORMAT,
        "--up-axis", EXPORT_UP_AXIS,
    ]

    if EXPORT_HALF_PRECISION:
        cmd.append("--half")

    log("EXPORT", "Exporting to USD ParticleField format")
    run_cmd(cmd, cwd=THREEDGRUT_DIR, stage="EXPORT")

    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        log("EXPORT", f"Done! Isaac Sim-ready file: {output_file} ({size_mb:.1f} MB)")
    else:
        log("EXPORT", f"Export completed but output file not found at {output_file}")
        usdz_files = list(EXPORT_DIR.glob("*.usdz")) + list(EXPORT_DIR.glob("*.usd"))
        if usdz_files:
            log("EXPORT", f"Found exported files: {[str(f) for f in usdz_files]}")


# ============================================================================
# MAIN
# ============================================================================
STAGES = {
    "1": ("COLMAP — Camera Pose Estimation", stage_1_colmap),
    "2": ("3DGRUT — 3D Gaussian Training", stage_2_train),
    "3": ("Export — USD for Isaac Sim", stage_3_export),
}


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]

    for s in requested:
        if s not in STAGES:
            print(f"Unknown stage: {s}")
            print(f"Usage: python {sys.argv[0]} [1] [2] [3]")
            print(f"  1 = COLMAP    2 = Train    3 = Export")
            sys.exit(1)

    print("=" * 60)
    print("  Images -> Isaac Sim 3D World (via 3DGRUT)")
    print(f"  Stages: {', '.join(requested)}")
    print("=" * 60)

    for s in requested:
        name, func = STAGES[s]
        print(f"\n{'=' * 60}")
        print(f"  Stage {s}: {name}")
        print(f"{'=' * 60}")
        func()

    print(f"\n{'=' * 60}")
    print("  COMPLETE")
    print("=" * 60)
    if "3" in requested:
        output_file = EXPORT_DIR / "scene.usdz"
        print(f"  Output:  {output_file}")
        print(f"  Load in Isaac Sim: File > Open > {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
