# Prologue: The Problem

There is a bin. Inside the bin there are objects. A robot arm needs to pick them up and place them somewhere else. For rigid, opaque, well-lit objects this has been solved for decades.

Then someone puts the objects in transparent plastic bags.

---

## What breaks

A modern industrial robot arm has no trouble moving to a precise 3D coordinate. Give it a position and an orientation and it will place its gripper there within a fraction of a millimeter, every time.

The hard part is computing those six numbers from a camera image.

The standard approach uses an RGB-D camera: a device that captures a color image and a depth image simultaneously. You find the object in the color image, read its depth, apply some geometry, and you have a 3D position.

Transparent packaging breaks both inputs at once.

**Color image:** a cookie in a transparent bag looks like whatever is behind it. The bag's surface produces only faint edge signals, no distinctive color, no recognizable texture. Classical detectors return nothing.

**Depth image:** structured-light depth cameras work by projecting an infrared dot pattern and measuring how it deforms. Transparent plastic transmits the infrared light instead of reflecting it. The sensor gets no return signal and writes zero into the depth buffer.

```
Color camera   sees: faint bag outline, contents barely visible
Depth camera    sees: zero at every pixel over the bag
Classical CV   finds: nothing
```

A real object, in a real bin, invisible to both sensing modalities.

---

## Why naive fixes fail

| Approach | Why it fails |
|----------|-------------|
| Swap to opaque packaging | Not always your decision |
| Add structured lighting | Reduces holes slightly, does not eliminate them |
| Use LiDAR instead | Expensive, impractical at bin-picking range |
| Rely on known bin geometry | Breaks when items shift or stack |

The real solution is a system that understands what it is looking at well enough to handle degraded sensing gracefully. That means combining classical geometry, 3D point clouds, and a trained deep learning model, each layer compensating for the others.

---

## What this book builds

Starting from: what is a pixel?

Ending at: a ROS 2 perception node publishing six-degree-of-freedom grasp poses for a cookie in a transparent bag.

The cookie is the running example throughout. Hard enough to be instructive, general enough that every technique applies to any similar transparent-packaging problem.

---

[Next: Chapter 1](01_how_computers_see.md)
