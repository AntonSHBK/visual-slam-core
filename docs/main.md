# Usage and Project Structure

This document describes the internal structure of the visual SLAM framework and outlines the recommended workflow for running, modifying, and extending the system. The project is primarily intended for research and experimental purposes; therefore, familiarity with visual odometry, SLAM concepts, and Python programming is assumed.

## Getting Started

```bash
pip install -r requirements.txt
```

The most convenient way to become familiar with the system is to explore the provided Jupyter notebooks:

```
notebooks/
└── VisualOdometry.ipynb
```

These notebooks demonstrate how to initialize the system, load datasets, run the SLAM pipeline, and visualize intermediate and final results. They are intended as an entry point for understanding the execution flow and for quick experimentation without writing standalone scripts.

## Configuration

All experiments start with the creation of a configuration object. The configuration defines camera parameters, feature extraction and matching settings, tracking thresholds, mapping options, and optimization parameters.

The configuration logic is centralized in:

```
visual_slam/config.py
```

The user is expected to explicitly construct and adjust the configuration before running the SLAM pipeline. This design choice makes all assumptions and parameters explicit and simplifies reproducibility of experiments.

## Data Sources

The framework supports different sources of visual input through a unified interface.

### Dataset-based Input

Pre-recorded datasets can be used via the dataset source module. A typical example is the KITTI Visual Odometry dataset.

A reference implementation and dataset structure compatible with this project can be found here:
[https://github.com/dawn-mathew/Visual-Odometry-KITTI-Sequence](https://github.com/dawn-mathew/Visual-Odometry-KITTI-Sequence)

The dataset source logic is implemented in:

```
visual_slam/source.py
```

This module is responsible for loading images, timestamps, calibration data, and feeding frames sequentially into the SLAM pipeline.

### Camera Input

Live camera input can also be used by implementing or extending the corresponding source class. The framework is designed such that any sensor providing sequential frames can be integrated with minimal changes, provided that camera intrinsics are properly defined.

## SLAM Processing Pipeline

Once the configuration and data source are defined, the SLAM process is executed through the main processing logic.

The recommended high-level workflow is as follows:

1. Create and configure the system parameters.
2. Initialize the SLAM system.
3. Sequentially feed frames into the tracking module.
4. Perform map initialization when sufficient parallax is observed.
5. Track camera motion frame-to-frame.
6. Insert keyframes when required.
7. Run local mapping and local optimization.
8. Visualize and log intermediate results.

This logic is encapsulated in:

```
visual_slam/processing.py
visual_slam/slam.py
```

Alternatively, advanced users may bypass the high-level processing wrapper and construct the pipeline manually, calling individual components directly for finer control and experimentation.

## Core Modules Overview

The framework is organized into modular subsystems, each responsible for a well-defined part of the SLAM pipeline.

### Feature Processing

```
visual_slam/feature/
```

This module includes feature detection, description, matching, and tracking. Users can implement custom feature extractors, matchers, or tracking strategies by extending the provided base classes.

### Tracking

```
visual_slam/tracking.py
visual_slam/trackingalgorithm/
```

Tracking handles frame-to-frame pose estimation, motion model usage, and failure detection. The current implementation focuses on monocular tracking but is structured to allow extension to other sensor modalities.

### Mapping and Map Representation

```
visual_slam/map/
visual_slam/local_mapping/
```

These modules define the internal map representation, including frames, keyframes, map points, and observations. Local mapping is responsible for triangulation, keyframe management, and maintaining local consistency.

### Optimization

```
visual_slam/optimization/
```

Optimization is implemented using PyTorch and operates directly on tensor-based representations of poses and map points. The framework allows the user to define custom optimization strategies by implementing new optimizer classes.

### Visualization

```
visual_slam/viz/
```

Visualization utilities provide tools for inspecting feature correspondences, camera trajectories, and reconstructed map structure. These components are primarily intended for debugging and qualitative analysis.

## Extensibility and Experimentation

A central goal of this framework is to support experimentation and algorithmic research. Most subsystems are designed around abstract base classes, allowing users to:

* Implement custom feature detection and matching pipelines
* Introduce alternative filtering or outlier rejection strategies
* Replace or extend tracking logic
* Develop new optimization methods and loss formulations
* Modify keyframe selection and local mapping policies

Users are strongly encouraged to explore the source code directly, as the framework is designed to be read, modified, and adapted rather than treated as a black-box SLAM solution.

## Notes

This project prioritizes transparency, modularity, and research flexibility over real-time performance. Users interested in understanding the internal mechanics of visual SLAM and experimenting with algorithmic variations are encouraged to work directly with the codebase and notebooks.
