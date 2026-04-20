# Teaching a Robot to See
### A practical guide to computer vision for robotic bin picking

---

A robot needs to pick objects from a bin. Some items come in transparent plastic bags. Transparent bags are nearly invisible to depth cameras, produce no useful color signal, and confuse every classical detector. Solving this requires the full modern computer vision stack: from a single pixel to a six-degree-of-freedom grasp pose delivered to a robot arm.

This book documents how that system was built. Every concept is introduced through the problem it solves. Every code snippet is there because the robot needed it.

---

## Who this is for

You write code. You are curious about robotics or computer vision. You want to understand not just what the algorithms are, but why they are shaped the way they are.

No robotics background required.

---

## Contents

| Chapter | Topic | Core question |
|---------|-------|---------------|
| [Prologue](chapters/00_prologue.md) | The problem | Why is this hard? |
| [1](chapters/01_how_computers_see.md) | How computers see | What is an image in memory? |
| [2](chapters/02_color_and_masking.md) | Color and masking | How do you isolate an object by color? |
| [3](chapters/03_geometry_from_shapes.md) | Geometry from shapes | How do you extract orientation from a silhouette? |
| [4](chapters/04_depth_and_backprojection.md) | Depth and back-projection | How do you turn a pixel into a 3D point? |
| [5](chapters/05_point_clouds.md) | Point clouds | How do you find objects in a 3D scene? |
| [6](chapters/06_limits_of_classical_cv.md) | When classical CV fails | Why is deep learning necessary here? |
| [7](chapters/07_yolo_and_segmentation.md) | YOLO and instance segmentation | How does a neural network detect objects? |
| [8](chapters/08_building_your_dataset.md) | Building your dataset | Why is data the most important decision? |
| [9](chapters/09_fine_tuning.md) | Fine-tuning | How do you teach a model your specific world? |
| [10](chapters/10_ros2_pipeline.md) | The ROS 2 pipeline | How does perception connect to a robot? |
| [11](chapters/11_six_dof_grasping.md) | Six degrees of freedom | How do you handle objects that are not lying flat? |
| [Epilogue](chapters/12_epilogue.md) | What this built | What did we learn, and where does it go? |

---

## Stack

```
Python 3.10    OpenCV 4.x    Open3D
YOLOv8-seg     ROS 2 Humble  Intel RealSense D435
NVIDIA GPU
```

---

## Interactive tools

Open these in a browser alongside the relevant chapter:

| Tool | Chapter | What it does |
|------|---------|-------------|
| [hsv_tuner.html](interactive/hsv_tuner.html) | 2 | Tune HSV thresholds live on synthetic scenes |
| [backprojection_explorer.html](interactive/backprojection_explorer.html) | 4 | See how pixels and depth become 3D points |
| [orientation_cluster_explorer.html](interactive/orientation_cluster_explorer.html) | 3, 5 | Explore minAreaRect angles and DBSCAN clustering |
