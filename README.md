# SatCorrect - Satellite Image Structure from Motion

SatCorrect is a tool for processing satellite imagery to reconstruct 3D structures using Structure from Motion (SfM) techniques. It provides camera parameter estimation, geometric correction, and bundle adjustment for satellite images.

## Features
- RPC to affine camera model conversion
- Skew and rotation correction for satellite images
- Multi-stage bundle adjustment
- Visualization tools for correspondence checking

## Dependencies
- Docker
- Python 3.7+
- GDAL
- PyProj
- NumPy
- SciPy
- OpenCV
- COLMAP

## Installation

1. Clone required repositories:
```bash
git clone https://github.com/Kai-46/ColmapForVisSat
```

2. Build Docker image:
```bash
docker build -f Dockerfile_for_ColmapForVisSat -t colmapforvissat:latest .
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python satellite_sfm.py --scene_path path/to/scene
```

### Parameters
- `--scene_path`: Path to scene directory containing input images
- `--center_crop`: Enable/disable center cropping (default: True)
- `--skew_correct`: Enable/disable skew correction (default: True) 
- `--rot_correct`: Enable/disable rotation correction (default: True)
- `--max_processes`: Number of parallel processes (default: -1 for auto)

### Input Structure
```
scene/
├── input/            # Input images
│   ├── IMG_0.tif
│   ├── IMG_1.tif
│   └── ...
├── scene_DSM.tif     # Digital Surface Model
├── scene_DSM.txt     # DSM metadata
└── ...
```

### Output Structure
```
scene/
├── images/           # Processed images
├── sparse/           # SfM results
│   ├── base/         # Initial reconstruction
│   │   ├── cameras.bin
│   │   ├── images.bin  
│   │   └── points3D.bin
│   └── 0/            # Final reconstruction
│       └── ...
├── cam_dict.json     # Camera parameters
└── ...
```

## Related Projects
- [SatelliteSfM](https://github.com/Kai-46/SatelliteSfM)
- [ColmapForVisSat](https://github.com/Kai-46/ColmapForVisSat) 
- [RAFT](https://github.com/princeton-vl/RAFT)
- [DFC2019](https://github.com/pubgeo/dfc2019/tree/master/track3)

## Examples

### Processing US3D dataset
```bash
python satellite_sfm.py --scene_path data/US3D_sample --max_processes 8
```

### Visualizing correspondences
```python
from utils.colmap_read_write_model import read_model
cameras, images, points = read_model("scene/sparse/base")
visualize_correspondence(images, cameras, points)
