# Neural Renderer

## Build

```bash
cmake -B build
cmake --build build --target viewer evaluate compare_images
```

With OptiX hardware ray tracing:

```bash
cmake -B build -DUSE_OPTIX=ON -DOPTIX_PTX_ARCH=compute_89
cmake --build build --target viewer
```

## Run

**Interactive viewer:**
```bash
./build/viewer <config.json>
```

**Render ground-truth vs neural (outputs to `comparison_output/`):**
```bash
./build/evaluate <config.json>
```

**Compare two images (PSNR + FLIP):**
```bash
./build/compare_images <reference.png> <test.png> [flip_output.png]
```

Scene configs are in `dbrt_data/<scene>/configs/`.
