# Epilogue: What This Built

---

## The full pipeline in one diagram

```
RGB image ----------> YOLO-seg detection ---------> pixel mask
                             |                          |
Depth image -----> hole ratio > 30%?                    |
                       yes -> use KNOWN_BOX_Z           |
                       no  -> patch median depth        |
                                    |                   |
                              pixel_to_3d <-- centroid from mask
                                    |
                              yaw from minAreaRect
                                    |
                              surface normal from depth mask
                              (fallback to vertical if < 50 pts)
                                    |
                              PoseStamped (camera frame)
                                    |
                              TF2 transform
                                    |
                              PoseStamped (base_link) -> robot arm
```

---

## Technical lessons

**Classical CV is complemented by deep learning, not replaced by it.** Color masking, contour analysis, and point cloud processing all earned their place. They run faster than any neural network, require no training data, and are trivially interpretable. The mistake is reaching for deep learning first. The right question is always: what is the simplest approach that solves this specific problem?

**The transparent packaging problem is a sensing problem, not a detection problem.** A neural network trained on enough examples detects the transparent bag reliably. What remains hard is the 3D position estimate, because the depth sensor genuinely cannot see it. The fallback depth and graceful tilt degradation are the correct architecture for a system that must work under degraded sensing.

**Data quality beats model size.** The choice between `yolov8s-seg` and `yolov8m-seg` had less impact on accuracy than the quality of the training labels. Relabeling the worst-performing images improved mAP50 more than switching to a larger model.

**The timestamp matters.** Using the original camera frame timestamp for TF2 lookups rather than the current time is a two-line difference that makes the system correct instead of subtly wrong. Details like this are invisible in tutorials and only appear when debugging a real robot.

---

## What comes next

**Force feedback.** The current system places the gripper at the computed pose and closes. A force-torque sensor at the wrist would detect partial grasps and trigger a retry. This eliminates the category of failure where the system is visually confident but mechanically unsuccessful.

**Active learning.** When the system falls back to the known-Z depth fallback it knows it is uncertain. That uncertainty signal can drive targeted data collection: automatically save the frames where fallback triggered, label them, retrain. The model gets better precisely where it is weakest.

**Multi-camera setup.** A second camera at a different angle eliminates depth holes for many bag orientations. The surface that is invisible from one angle is opaque from another. The architectural changes are modest: subscribe to two synchronized camera pairs, fuse the resulting poses.

---

## On building with imperfect sensing

The cookie in a transparent bag is never going to be perfectly visible. The depth holes are real. The color signal is genuinely absent. The system described in this book does not solve those physical facts. It works around them gracefully.

That is, ultimately, what robust engineering is: not eliminating uncertainty, but building systems that behave well in its presence.

---

[Prev: Chapter 11](11_six_dof_grasping.md) | [Back to contents](../README.md)
