#!/usr/bin/env python3
"""
build_nurec_world.py — NuRec Neural Reconstruction for Isaac Sim 5.1

Uses NVIDIA 3DGUT (Gaussian splatting with MCMC densification) to create
photorealistic USD scenes from drone imagery. Based on:
  https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html

Pipeline:
  Stage 1: COLMAP SfM   — sparse reconstruction with PINHOLE model
  Stage 2: 3DGUT Train  — neural Gaussian splatting (MCMC, 30K iterations)
  Stage 3: USD Export    — ParticleField3DGaussianSplat for Isaac Sim 5.1

Usage:
  python build_nurec_world.py --season summer              # all stages
  python build_nurec_world.py --season summer 1            # COLMAP only
  python build_nurec_world.py --season summer 2 3          # train + export
  python build_nurec_world.py --season summer --iterations 50000  # longer training

Prerequisites:
  - COLMAP with CUDA at /usr/local/bin/colmap
  - 3dgrut venv active (conda activate 3dgrut)
  - NVIDIA GPU with ≥16GB VRAM (24GB recommended)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

COLMAP_BIN = "/usr/local/bin/colmap"
COLMAP_CAMERA_MODEL = "PINHOLE"
COLMAP_MAX_IMAGE_SIZE = 3200
COLMAP_USE_GPU = True

THREEDGRUT_DIR = Path(__file__).parent / "3dgrut"
TRAIN_SCRIPT = THREEDGRUT_DIR / "train.py"
TRAIN_CONFIG = "apps/colmap_3dgut_mcmc.yaml"
DEFAULT_ITERATIONS = 30000

EXPORT_USD_FORMAT = "nurec"
EXPORT_UP_AXIS = "z"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ============================================================================
# DERIVED PATHS
# ============================================================================

INPUT_IMAGES_DIR: Path = None  # type: ignore[assignment]
OUTPUT_DIR: Path = None  # type: ignore[assignment]
COLMAP_WS: Path = None  # type: ignore[assignment]
TRAIN_OUT: Path = None  # type: ignore[assignment]
EXPORT_DIR: Path = None  # type: ignore[assignment]


def setup_paths(season):
    global INPUT_IMAGES_DIR, OUTPUT_DIR, COLMAP_WS, TRAIN_OUT, EXPORT_DIR
    base = Path(__file__).parent / "rusko"
    INPUT_IMAGES_DIR = base / f"rusko_{season}"
    OUTPUT_DIR = base / f"output_{season}_nurec"
    COLMAP_WS = OUTPUT_DIR / "colmap_workspace"
    TRAIN_OUT = OUTPUT_DIR / "3dgut_train"
    EXPORT_DIR = OUTPUT_DIR / "export"


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


# ============================================================================
# STAGE 1: COLMAP Sparse Reconstruction (PINHOLE model for 3DGUT)
# ============================================================================

def stage_1_colmap():
    """Run COLMAP SfM with PINHOLE camera model as required by 3DGUT."""
    images = sorted(
        p for p in INPUT_IMAGES_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        log("SFM", f"No images found in {INPUT_IMAGES_DIR}")
        sys.exit(1)
    log("SFM", f"Found {len(images)} images in {INPUT_IMAGES_DIR}")

    COLMAP_WS.mkdir(parents=True, exist_ok=True)

    images_dir = COLMAP_WS / "images"
    if not images_dir.exists():
        images_dir.symlink_to(INPUT_IMAGES_DIR.resolve())
        log("SFM", f"Symlinked images → {INPUT_IMAGES_DIR}")

    db_path = COLMAP_WS / "database.db"
    sparse_dir = COLMAP_WS / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if COLMAP_USE_GPU else "0"

    # Feature extraction
    log("SFM", "Extracting features (SIFT)")
    run_colmap([
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", COLMAP_CAMERA_MODEL,
        "--SiftExtraction.use_gpu", gpu_flag,
        "--SiftExtraction.max_image_size", str(COLMAP_MAX_IMAGE_SIZE),
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ])

    # Feature matching — use spatial if GPS available, else exhaustive
    log("SFM", "Matching features (exhaustive)")
    run_colmap([
        "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ])

    # Sparse reconstruction
    log("SFM", "Running mapper (sparse reconstruction)")
    run_colmap([
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ])

    sparse_0 = sparse_dir / "0"
    if not sparse_0.exists():
        log("SFM", "ERROR: Sparse reconstruction failed — no model produced.")
        sys.exit(1)

    n_images = count_colmap_images(sparse_0)
    log("SFM", f"Sparse reconstruction complete: {n_images} registered images")


def run_colmap(args):
    cmd = [COLMAP_BIN] + args
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log("SFM", f"COLMAP failed: {' '.join(args[:1])}")
        sys.exit(1)


def count_colmap_images(sparse_dir):
    import struct
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        return 0
    with open(images_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


# ============================================================================
# STAGE 2: 3DGUT Training (Gaussian Splatting with MCMC)
# ============================================================================

def stage_2_train(iterations=DEFAULT_ITERATIONS):
    """Train 3DGUT model on the COLMAP sparse reconstruction."""
    sparse_0 = COLMAP_WS / "sparse" / "0"

    # Check for existing COLMAP data — also accept from build_mesh_world.py
    if not sparse_0.exists():
        alt_colmap = Path(__file__).parent / "rusko" / f"output_{INPUT_IMAGES_DIR.name.replace('rusko_', '')}" / "colmap_workspace"
        if (alt_colmap / "sparse" / "0").exists():
            log("TRAIN", f"No NuRec COLMAP found. Reusing existing at {alt_colmap}")
            COLMAP_WS.mkdir(parents=True, exist_ok=True)

            for item in ["sparse", "database.db"]:
                src = alt_colmap / item
                dst = COLMAP_WS / item
                if src.exists() and not dst.exists():
                    dst.symlink_to(src.resolve())

            images_link = COLMAP_WS / "images"
            if not images_link.exists():
                images_link.symlink_to(INPUT_IMAGES_DIR.resolve())
        else:
            log("TRAIN", "No COLMAP sparse reconstruction found. Run stage 1 first.")
            sys.exit(1)

    if not (COLMAP_WS / "images").exists():
        images_link = COLMAP_WS / "images"
        images_link.symlink_to(INPUT_IMAGES_DIR.resolve())

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)

    log("TRAIN", f"Starting 3DGUT MCMC training ({iterations} iterations)")
    log("TRAIN", f"  COLMAP path: {COLMAP_WS}")
    log("TRAIN", f"  Output: {TRAIN_OUT}")
    log("TRAIN", f"  Config: {TRAIN_CONFIG}")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config-name", TRAIN_CONFIG,
        f"path={COLMAP_WS}",
        f"out_dir={TRAIN_OUT}",
        "experiment_name=nurec_mcmc",
        f"n_iterations={iterations}",
        "export_usd.enabled=true",
        "export_usd.apply_normalizing_transform=true",
        f"export_usd.format={EXPORT_USD_FORMAT}",
    ]

    log("TRAIN", f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(THREEDGRUT_DIR))
    if result.returncode != 0:
        log("TRAIN", "Training failed. Check output above for errors.")
        sys.exit(1)

    log("TRAIN", "Training complete.")


# ============================================================================
# STAGE 3: USD Export for Isaac Sim 5.1
# ============================================================================

def stage_3_export():
    """Export trained model to USD for Isaac Sim 5.1."""
    train_dir = TRAIN_OUT / "nurec_mcmc"

    # Find the checkpoint
    ckpt_path = train_dir / "ckpt_last.pt"
    if not ckpt_path.exists():
        ckpt_candidates = sorted(train_dir.glob("ckpt_*.pt"))
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[-1]
        else:
            log("USD", f"No checkpoint found in {train_dir}")
            sys.exit(1)

    log("USD", f"Using checkpoint: {ckpt_path}")

    # Check if USD was already exported during training (trainer writes .usdz by default)
    usd_files = sorted(train_dir.glob("export_last_*.usdz")) + sorted(train_dir.glob("export_last_*.usd"))
    if usd_files:
        log("USD", f"Found export from training: {usd_files[-1]}")
        usdz_src = usd_files[-1]
    else:
        # Run standalone export as plain .usd (not .usdz — better Isaac Sim 5.1 compat)
        log("USD", "Exporting USD from checkpoint")
        usdz_src = train_dir / "export_nurec.usd"
        cmd = [
            sys.executable, "-m", "threedgrut.export.scripts.export_usd",
            "--checkpoint", str(ckpt_path),
            "--output", str(usdz_src),
            "--format", EXPORT_USD_FORMAT,
            "--up-axis", EXPORT_UP_AXIS,
            "--dataset", str(COLMAP_WS),
            "--verbose",
        ]
        log("USD", f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(THREEDGRUT_DIR))
        if result.returncode != 0:
            log("USD", "Export failed.")
            sys.exit(1)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    season = INPUT_IMAGES_DIR.name.replace("rusko_", "")
    ext = usdz_src.suffix
    dest = EXPORT_DIR / f"rusko_{season}_nurec{ext}"
    shutil.copy2(str(usdz_src), str(dest))
    log("USD", f"Exported: {dest}")
    log("USD", f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")

    # If training produced .usdz, also do standalone .usd export for Isaac Sim 5.1 compat
    if ext == ".usdz" and ckpt_path.exists():
        log("USD", "Also exporting as plain .usd (better Isaac Sim 5.1 compatibility)")
        usd_dest = EXPORT_DIR / f"rusko_{season}_nurec.usd"
        cmd = [
            sys.executable, "-m", "threedgrut.export.scripts.export_usd",
            "--checkpoint", str(ckpt_path),
            "--output", str(usd_dest),
            "--format", EXPORT_USD_FORMAT,
            "--up-axis", EXPORT_UP_AXIS,
            "--dataset", str(COLMAP_WS),
        ]
        result = subprocess.run(cmd, cwd=str(THREEDGRUT_DIR))
        if result.returncode == 0:
            log("USD", f"Exported: {usd_dest} ({usd_dest.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            log("USD", "Plain .usd export failed — use the .usdz file instead")

    log("USD", "")
    log("USD", "To use in Isaac Sim 5.1:")
    log("USD", "  1. File → Import → select the .usd file")
    log("USD", "  2. Create → Physics → Ground Plane (align with scene floor)")
    log("USD", "  3. Connect proxy mesh for collision (see NuRec docs)")
    log("USD", "  4. Create → Robots → add your drone")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NuRec Neural Reconstruction → Isaac Sim 5.1 USD"
    )
    parser.add_argument("--season", required=True, choices=["summer", "winter"])
    parser.add_argument("stages", nargs="*", type=int, default=[1, 2, 3],
                        help="Stages to run (1=COLMAP, 2=Train, 3=Export)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Training iterations (default: {DEFAULT_ITERATIONS})")
    args = parser.parse_args()

    setup_paths(args.season)

    stages_to_run = args.stages if args.stages else [1, 2, 3]

    print("=" * 60)
    print("  NuRec Neural Reconstruction → Isaac Sim 5.1 USD")
    print(f"  Season: {args.season}")
    print(f"  Stages: {', '.join(map(str, stages_to_run))}")
    print(f"  Iterations: {args.iterations}")
    print("=" * 60)
    print()

    stage_map = {
        1: ("COLMAP Sparse Reconstruction (PINHOLE)", stage_1_colmap),
        2: ("3DGUT MCMC Training", lambda: stage_2_train(args.iterations)),
        3: ("USD Export for Isaac Sim 5.1", stage_3_export),
    }

    for s in stages_to_run:
        if s not in stage_map:
            print(f"Unknown stage: {s}")
            sys.exit(1)
        title, func = stage_map[s]
        print("=" * 60)
        print(f"  Stage {s}: {title}")
        print("=" * 60)
        func()
        print()

    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
