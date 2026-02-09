# CUDA Mesh Renderer

CUDA-accelerated mesh renderer with interactive viewer and evaluation tools. Supports both classic path tracing and shell-based neural rendering for real-time visualization and offline quality comparison.

## Executables

### imgui_renderer
Interactive viewer with ImGui interface for real-time rendering and camera exploration.

### evaluate
Headless comparison tool that renders ground truth and neural images, computing PSNR and FLIP perceptual error metrics.

### compare_images
Standalone tool to compute PSNR and FLIP metrics between two existing images.

## Configuration

Both imgui_renderer and evaluate use JSON configuration files to specify scene setup, rendering parameters, and neural network settings. Example configs are provided in the [configs/](configs/) directory.

Key configuration options:
- **Scene**: Paths to original mesh, inner/outer shells, and optional additional mesh with scale factors
- **Checkpoint**: Path to trained neural network weights (.bin file)
- **Environment**: HDRI path, rotation angle, and strength multiplier
- **Camera**: 4x4 transformation matrix (column-major) and vertical field of view
- **Material**: Disney BRDF parameters (base color, roughness, metallic, specular, etc.)
- **Neural Network**: Hash map size and neural query toggle
- **Rendering**: Mesh normalization and texture sampling settings

See [configs/statuette_obj.json](configs/statuette_obj.json) for a complete example.

## Build

Requirements:
- CMake 3.21+
- CUDA toolkit (nvcc) with compute capability 8.0+
- OpenGL development libraries

Build steps:
```
cmake -S . -B build
cmake --build build -j
```

On first configure, CMake will fetch dependencies from GitHub:
- GLFW, ImGui (UI framework)
- Assimp, TinyGLTF (mesh loading)
- BVH (acceleration structure)
- tiny-cuda-nn (neural network inference)
- TinyEXR, STB (image I/O)
- FLIP (perceptual error metric)
- nativefiledialog-extended (file picker)

## Usage

### Interactive Renderer

Run with default config:
```
./build/imgui_renderer
```

Run with custom config:
```
./build/imgui_renderer /path/to/config.json
```

The default config is [configs/chess.json](configs/chess.json). If no mesh is specified in the config or loading fails, a procedural UV sphere is generated.

Supported mesh formats:
- GLTF/GLB (with full material and texture support)
- OBJ, FBX (geometry only, uses global material)

Controls:
- `WASD`: move camera
- `Q/E`: move up/down
- Mouse drag: look around
- `ESC`: release mouse
- Left click: recapture mouse

Features:
- Real-time path tracing with adjustable samples per pixel
- Neural query mode for shell-based intersection prediction
- Camera export to JSON (generates timestamped filename)
- File picker for loading meshes at runtime
- Lambert shading toggle
- Bounce count control
- Material parameter adjustment (base color, roughness, metallic, etc.)
- HDRI environment lighting with rotation and strength controls

### Evaluation Tool

The evaluate tool performs offline comparison between ground truth (classic path tracing) and neural rendering.

Run with default config:
```
./build/evaluate
```

Run with custom config:
```
./build/evaluate /path/to/config.json
```

The default config is [configs/statuette_obj.json](configs/statuette_obj.json).

Configuration format (JSON):
```json
{
  "scene": {
    "original_mesh": {"path": "/path/to/mesh.obj", "scale": 1.0},
    "inner_shell": {"path": "/path/to/inner.obj", "scale": 1.0},
    "outer_shell": {"path": "/path/to/outer.obj", "scale": 1.0}
  },
  "checkpoint_path": "/path/to/checkpoint.bin",
  "environment": {
    "hdri_path": "/path/to/environment.hdr",
    "rotation": 0.0,
    "strength": 1.0
  },
  "camera": {
    "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "yfov": 0.7854
  },
  "material": {
    "base_color": [1.0, 1.0, 1.0],
    "roughness": 1.0,
    "metallic": 0.0
  },
  "neural_network": {
    "log2_hashmap_size": 14,
    "use_neural_query": false
  }
}
```

Rendering parameters are set in [src/main_comparison.cu](src/main_comparison.cu:24-28):
- Resolution: 1920x1080
- Samples: 2048 (batched in groups of 4)
- Bounce count: 3

Output:
- `comparison_output/ground_truth.png` - Classic path tracing result
- `comparison_output/neural.png` - Neural rendering result
- `comparison_output/flip_error.png` - FLIP perceptual error heatmap (Magma colormap)
- Console output with PSNR and mean FLIP metrics

The tool uses batched rendering to handle high sample counts without GPU memory overflow.

### Image Comparison Tool

Computes PSNR and FLIP perceptual error metrics between two existing images.

Usage:
```
./build/compare_images <reference_image> <test_image> [flip_output.png]
```

Examples:
```bash
# Compute metrics only
./build/compare_images ground_truth.png neural.png

# Compute metrics and save FLIP visualization
./build/compare_images ground_truth.png neural.png flip_error.png
```

Supported formats: PNG, JPG, BMP, TGA, and other formats supported by stb_image.

Output:
- PSNR value in dB (higher is better, >30 dB is generally good quality)
- FLIP mean error (lower is better, 0.0 = identical)
- Optional FLIP error heatmap visualization (Magma colormap: black = no error, violet = high error)
