# Roadmap

## Current State (June 2026)

- [x] Drone images acquired (summer + winter, ~700 each)
- [x] 3DGS pipeline (`build_3d_world.py`) — written but blocked by Isaac Sim 5.1 lacking 3DGS support
- [x] Mesh pipeline (`build_mesh_world.py`) — written, ready to run
- [x] COLMAP 3.9.1 built from source with CUDA
- [x] All Python dependencies installed (xatlas, Open3D, OpenCV, OpenUSD, scipy)
- [ ] **Run mesh pipeline on summer images**
- [ ] **Run mesh pipeline on winter images**
- [ ] Validate USD output in Isaac Sim 5.1
- [ ] Test with Pegasus Simulator drone

## Next Steps

### Phase 1: Generate 3D Models (immediate)

1. Run `python build_mesh_world.py --season summer` (~5-7 hrs)
2. Inspect intermediate outputs:
   - `fused.ply` in MeshLab/CloudCompare — verify dense coverage
   - `visual_mesh.ply` — check for holes, floating debris
   - `texture_atlas.png` — check for blurring, seams, color artifacts
3. Load `rusko_summer.usd` in Isaac Sim 5.1 — verify textures render
4. Repeat for winter

### Phase 2: Quality Improvements (if needed)

- **Texture seams**: Increase `TEXTURE_ATLAS_SIZE` to 8192, or tile scene into multiple USD prims with separate atlases
- **Geometric holes**: Increase `POISSON_DEPTH` (12-13), or use COLMAP's `delaunay_mesher` instead of Poisson
- **Mesh artifacts under overhangs**: Poisson tends to fill concavities — consider screened Poisson or alpha-shape trimming
- **Scale/georeferencing**: Run `colmap model_aligner` after Stage 1 to align reconstruction to GPS (metric scale + orientation)
- **LOD**: Generate multiple mesh resolutions for Isaac Sim performance at different drone altitudes

### Phase 3: Simulation Integration

- Configure Pegasus Simulator world loading with the USD files
- Add semantic labels (buildings, roads, vegetation) if needed for perception tasks
- Set up LiDAR sensor simulation using the collision mesh
- Test visual navigation / obstacle avoidance

### Phase 4: Future — Isaac Sim 6.0 Migration

When Pegasus Simulator supports Isaac Sim 6.0:
- Switch to `build_3d_world.py` (3DGS pipeline) for higher visual quality
- Use ParticleField USDZ format (native 3DGS rendering)
- Keep mesh pipeline for collision geometry

## Known Issues

- **COLMAP system binary** (`/usr/bin/colmap`) has no CUDA — do NOT use it for MVS.
  Always use `/usr/local/bin/colmap` (3.9.1 with CUDA).
- **COLMAP source** (`colmap/`) is checked out at tag `3.9.1` (detached HEAD).
  The `main` branch requires faiss + CMake 3.24+ which this system doesn't have.
- **pycolmap** pip package (4.0.4) is installed but `has_cuda=False` — cannot be
  used for dense MVS. Only useful for SfM Python API.
- Texture baking quality depends on camera coverage. Areas seen from only 1-2
  cameras at oblique angles will have lower texture resolution.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Mesh pipeline over 3DGS | Isaac Sim 5.1 can't render 3DGS; Pegasus doesn't support 6.0 |
| COLMAP 3.9.1 over latest | Latest requires faiss + CMake 3.24; 3.9.1 has full MVS/CUDA |
| Spatial matching over exhaustive | GPS-tagged drone images; spatial is 10-100x faster |
| Multi-view texture projection | User chose maximum photorealism over simpler vertex colors |
| Separate scripts (not refactored) | Different target platforms; keep both for future flexibility |
| UsdPreviewSurface (not OmniPBR) | Universal USD material, works in headless Isaac Sim |
| 8K texture atlas | Balances detail vs file size for neighborhood-scale scene |
