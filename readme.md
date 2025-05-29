# GMatch - a feature matcher for RGBD-based 6DoF pose estimation

## 1. Introduction


This repository implements GMatch in `gmatch.py` and provides demo code of solving pose from correspondence. The pose estimation pipeline is:

**`keypoint descriptor` => `feature matcher` => `geometric solver` (=> `refiner`)**

In this repository, the key part is `gmatch.search()`, which searchs geometrically consistent correspondence in 3D keypoints. It is the python implementation of the pseudo code we provide in the paper and our core innovation.
Besides, there are some wrap-up or visualization functions that you may find handy.
- `gmatch.Match()` is the wrap-up of `gmatch.search()` to match each of source images to the target image for pose estimation
- `util.Solve()` demonstrates how to use Kabsch solver, RANSAC and ICP refiner to get the object's 6DoF pose w.r.t camera coordinate system in the scene image.
- `util.plot_keypoints()` to visualize keypoints and mark locally matched points interactively under given `thresh_feat`, which is helpful when adapting GMatch to new keypoint descriptors.


## 2. Installation

To install dependency of GMatch:
```
pip install -r requirements.txt
```

## 3. Usage
We provide `demo.py` for demonstration purpose and you can run it with `python demo.py`.

> Note: To understand how to use GMatch for pose estimation, we strongly recommend you to read `demo.py` line by line.

In brief, it first renders from six views to get images and point clouds as source, followed by loading RGB-D images as target. Then it formats the data to put them into `MatchData` and pass this to `gmatch.Match()`. Finally, with the match result stored in `MatchData`, `util.Solve()` gives the estimated object pose w.r.t scene camera coordinate system.

We also put some visualization codes into `demo.py` so that you can see how each part goes.


## 4. Reproduction of Paper Results
To get HOPE and YCB-Video datasets on BOP platform, follow the instructions in https://bop.felk.cvut.cz/datasets/. Since our method is zero-shot, training images can be ignored safely. After download and unzip, you should get files like that:
```
.
├── dataset_info.md
├── models/
├── models_eval/
├── test/
├── test_targets_bop19.json
...
```
After that, create a directory named `bop_data` and put datasets there such that `hope` can be accessed via `./bop_data/hope`.
To test different combination of NN/GMatch/LightGlue and SIFT/ORB/SuperPoint, use `git checkout <TAG_NAME>` and `python main.py`. Available tags can be listed by `git tag -l`.

> Note: To play with LightGlue or SuperPoint, you have to install lightglue as in https://github.com/cvg/LightGlue.git

> Note: To test GMatch-ORB, you should not only `git checkout GMatch-ORB`, but also uncomment line 27~33 and comment line 16~23 in gmatch.py.


## Tips
- To render depth (e.g. `get_snapshot()`), it's recommended to set `LIBGL_ALWAYS_SOFTWARE` to `1`. In RTX4060 + Ubuntu20.04-WSL, Hardware rendering leads to abnormal depth.
- Codes below """ visualization """ or """ \<Tune\> xxx """ can be safely commented/uncommented to see how's the program going.
