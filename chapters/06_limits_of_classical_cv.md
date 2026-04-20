# Chapter 6: When Classical CV Fails

Chapters 2 through 5 built a complete perception system using color thresholding, contour analysis, and 3D point cloud processing. For opaque objects with distinctive color or shape this system is fast, reliable, and needs no training data.

Then you point it at a cookie in a transparent plastic bag.

---

## The three failure modes

**Color masking fails completely.** The transparent bag has no distinctive color. The cookie inside is partially visible through the plastic, inconsistently so depending on lighting angle and bag orientation. There is no HSV range to threshold on.

**Depth masking fails for the object.** Every zero pixel over the transparent bag is a genuine sensor failure: the infrared light passed through. The fallback depth gives an approximate Z. It says nothing about exact position within the bin, orientation, or stacking.

**Point cloud clustering finds a hole.** After floor removal, the region under the transparent bag appears as a gap in the cloud: floor points are missing because the bag blocks them, but the bag contributes no points either. DBSCAN finds nothing.

The classical pipeline returns silence.

---

## What changes at the fundamental level

Classical CV computes hand-crafted features: color histograms, gradient magnitudes, connected components. These features assume objects have distinctive colors, reflect light predictably, and occupy contiguous regions in depth.

Transparent packaging violates all three assumptions.

Deep learning learns directly from examples. Given enough labeled images of the transparent bag, the network learns whatever statistical patterns distinguish it from background: the refraction pattern at the bag edge, the subtle shadow it casts, the slight change in background texture visible through the plastic. It does not know it is looking for "a transparent bag." It learns features that correlate with the human labels.

---

## What you need from a detector

For bin picking specifically:

1. **A pixel-accurate mask** per instance, not a bounding box. The centroid of a bounding box falls on empty space when an elongated bag is rotated. The centroid of a pixel mask is always inside the object.
2. **Per-instance confidence scores**, so you can filter uncertain detections.
3. **Class labels**, to distinguish different object types.
4. **Runtime under 100ms** on a GPU, for a real-time robot control loop.

This combination points to **instance segmentation**: a network that produces a separate pixel mask for each detected object instance.

---

## Why instance segmentation specifically

Instance segmentation separates individual objects even when they are touching, provides mask centroids that are always geometrically valid, and gives you the object's shape directly, from which orientation is a single `minAreaRect` call away.

This is the same analysis as Chapters 2-3, now applied to a learned mask instead of a color-thresholded one.

---

## What deep learning costs

| Requirement | Classical CV | Deep learning |
|-------------|-------------|---------------|
| Training data | None | 80-200 labeled images per class |
| GPU | Optional | Required for real-time |
| Interpretability | High | Lower |
| Generalisation | Poor for transparent objects | Strong if data is diverse |

Chapter 7 covers the YOLO architecture. Chapters 8 and 9 cover building the dataset and fine-tuning. The result is a model that handles what five chapters of classical CV could not.

---

[Prev: Chapter 5](05_point_clouds.md) | [Next: Chapter 7](07_yolo_and_segmentation.md)
