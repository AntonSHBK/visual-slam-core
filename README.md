# Visual SLAM Framework (Python)

<img src="docs/media/Матчи%20инициализация.jpg" alt="Match" width="600"/>  

## Overview

This repository contains a research-oriented implementation of a visual SLAM (Simultaneous Localization and Mapping) system developed as part of a dissertation and ongoing scientific research in the field of computer vision and autonomous navigation. The project focuses on the design, implementation, and experimental analysis of a modular visual SLAM pipeline, with an emphasis on clarity of architecture, extensibility, and reproducibility of results.

The system is designed as a fully Python-based visual SLAM framework, targeting monocular camera input, with a clear separation between initialization, tracking, mapping, and optimization components. The primary application domain includes UAVs and mobile robotic platforms operating in unknown environments.

## Motivation and Background

The development of this framework was inspired by established and widely used SLAM systems, in particular:

* **pySLAM**
  [https://github.com/luigifreda/pyslam](https://github.com/luigifreda/pyslam)

* **ORB-SLAM3**
  [https://github.com/UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

These frameworks served as conceptual and architectural references, especially in terms of pipeline decomposition, map representation, and optimization strategies. However, this project does not aim to replicate their implementations, but rather to explore an alternative design that prioritizes Python-based development and research flexibility.

**Finish point cloud**

<img src="docs/media/map_1.jpg" alt="Map 1>" width="600"/>  

<img src="docs/media/map_2.jpg" alt="Map 2" width="600"/>  

<img src="docs/media/map_3.jpg" alt="Map 3" width="600"/>  

## Project Status

**Work in Progress**

This project is currently under active development and should be considered experimental.

At the present stage, the following components have been implemented and integrated:

* Map initialization for monocular input
* Frame-to-frame tracking
* Local mapping and keyframe handling
* Local optimization of poses and map points
* Internal map representation (frames, keyframes, map points, observations)
* Visualization of the map, camera trajectory, and feature correspondences
* Logging and debugging utilities for core subsystems

Global optimization, loop closing, and large-scale consistency mechanisms are not yet fully implemented and remain part of future work.

## Key Characteristics

A defining feature of this framework is its **independence from compiled third-party SLAM libraries**.

* The entire system is implemented in **pure Python**
* No mandatory C++ extensions or external SLAM backends are required
* Optimization is implemented using **PyTorch**, operating directly on tensor representations
* This design choice simplifies experimentation, debugging, and modification of optimization strategies at the cost of computational efficiency

As a result, the current implementation is not optimized for real-time performance and is significantly less efficient than mature C++-based SLAM systems. Performance improvements, algorithmic refinements, and possible hybrid acceleration strategies are considered future research directions.

## Usage

Detailed instructions on how to run and experiment with the system are provided in a separate documentation file:

* **Usage and execution guide:**
  [`docs/main.md`](docs/main.md)

This document describes dataset preparation, configuration, execution flow, and available visualization and logging options.

<!-- ## Citation

```bibtex
@misc{px4_multi_drone_sim,
  author       = {Anton Pisarenko},
  title        = {PX4 Multi-Drone Simulation Project},
  year         = {2024},
  month        = {December},
  howpublished = {\url{https://github.com/AntonSHBK/px4_multi_drone_sim}},
  note         = {Accessed: 2024-12-24}
}
``` -->

## Contacts

For questions or support, please contact:

- **Name**: Pisarenko Anton
- **Email**: anton42@yandex.ru
- **Telegram**: [antonSHBK](https://t.me/antonSHBK)





