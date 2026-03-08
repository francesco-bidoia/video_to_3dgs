# Video/Image Folder to 3D Gaussian Splatting

This repository provides an end-to-end pipeline for generating 3D Gaussian Splatting models from a scene folder. The scene can contain either a video file or an `images/` folder.

## Overview

The pipeline currently supports:
1. Input detection from `source_path` (video or `images/`)
2. Structure from Motion (SfM) using COLMAP
3. Image undistortion to COLMAP format (`undistorted/`)
4. 3D Gaussian Splatting training

## Installation

### Prerequisites
- Docker
- NVIDIA GPU with CUDA support
- WSL2 (if using Windows) or Ubuntu 22.04

### Setup

1. Clone this repository:
   ```bash
   git clone --recursive https://github.com/yourusername/video_to_3dgs.git
   cd video_to_3dgs
   ```

2. Download Depth Anything V2 model:
   - Download the Depth-Anything-V2-Large model from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
   - Place it in: `submodules/DepthAnythingV2_docker/checkpoints/depth_anything_v2_vitl.pth`

3. Build and start the Docker container:
   ```bash
   ./start.sh [--dataset=/path/to/dataset_gs]
   ```
   This script will:
   - Automatically build the Docker container with all required dependencies
   - Detect your environment (WSL1, WSL2, or native Linux) and configure X11 forwarding
   - Use the default dataset path (`../datasets_gs`) or a custom path specified with `--dataset`

## Usage

### Scene Preparation

Create a scene directory in one of these two formats:

1. Video input:
   ```
   datasets_gs/your_scene_name/your_video.mp4
   ```

2. Image-folder input:
   ```
   datasets_gs/your_scene_name/images/
   ```
   Put your source images directly in `images/`.

Input priority rule:
- If both a video and `images/` exist, the pipeline assumes the video was already processed and uses `images/`.

   By default, the pipeline looks for datasets in the `../datasets_gs` directory relative to the project root. You can specify a different location when starting the container:
   ```bash
   ./start.sh --dataset=/path/to/your/datasets
   ```

### Running the Pipeline

To process a single scene:
```bash
python do_all.py -s /v2gs/datasets_gs/your_scene_name -n 300
```

Parameters:
- `-s, --source_path`: Path to the scene directory (must contain either `images/` or a video file)
- `-n, --max_number_of_frames`: Maximum number of frames to extract (default: 400)
- `-c, --clean`: Clean existing processed data and start fresh
- `-m, --minimal`: Use minimal frame selection after final reconstruction
- `-f, --full`: Use all frames for reconstruction and longer training
- `--full_res`: Extract final frames at full resolution
- `-a, --all`: Process all subdirectories in the source path

Resume behavior:
- The pipeline checks existing outputs and continues from what is already available.
- If `undistorted/` already exists and is complete, undistortion is skipped.
- `--clean` removes generated artifacts (`tmp/`, `sparse/`, `undistorted/`, DB files, model outputs) and keeps raw scene inputs.

### Output

The pipeline generates these key artifacts under `source_path`:
- `images/`: input image set (pre-existing or extracted from video)
- `database.db` and `database_final.db`: COLMAP databases
- `sparse/0/`: final sparse reconstruction
- `undistorted/images/` and `undistorted/sparse/0/`: output of `colmap image_undistorter`
- `model/`: trained 3D Gaussian Splatting model

## Viewing Results

Use the webtool: [supersplat](https://superspl.at/editor)
Simply drag and drop the .ply file in the browser

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses the following open-source projects:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [COLMAP](https://github.com/colmap/colmap)
