# GMatch - a feature matcher for RGBD-based 6DoF pose estimation

## 1. Introduction


This repository implements GMatch in `gmatch.py` and provides demo code of solving pose from correspondence. The pose estimation pipeline is:

**`keypoint descriptor` => `feature matcher` => `geometric solver` (=> `refiner`)**

In this repository, the key part is `gmatch.Search()`, which searchs geometrically consistent correspondence in 3D keypoints. It is the python implementation of the pseudo code we provide in the paper and our core innovation.
Besides, there are some wrap-up or visualization functions that you may find handy.
- `gmatch.Match()` is the wrap-up of `gmatch.Search()` to match each of source images to the target image for pose estimation
- `main.solve()` demonstrates how to use Kabsch solver, RANSAC and ICP refiner to get the object's 6DoF pose w.r.t camera coordinate system in the scene image.
- `main.render()` use open3d to render 6 views of CAD model. we call the result `snapshots`, which combines RGB images, reconstructed point clouds from depth images, camera poses and masks.
- `util.py` to visualize snapshots, keypoints (useful for determine feature similarity threshold) and final matches.



## 2. Installation and Usage

To install dependency of GMatch:
```
pip install -r requirements.txt
```

To understand how to use GMatch for pose estimation, check the `demo.py`, which estimate the sugar box's pose with following command. (scene images and obj model are in `sugar_box/`)
```
python demo.py
```


## 3. Reproduction of Paper Results
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
After that, create a directory named `bop_data` and put datasets there such that `main.py` can find `hope` with `./bop_data/hope`.
To test different combination of NN/GMatch/LightGlue and SIFT/ORB/SuperPoint, use `git checkout <TAG_NAME>` and `python main.py`. Available tags can be listed by `git tag -l`.

> Note: To play with LightGlue or SuperPoint, you have to install lightglue as in https://github.com/cvg/LightGlue.git

> Note: To test GMatch-ORB, in addition to `git checkout GMatch-ORB`, uncomment line 27~33 and comment line 16~23 in gmatch.py.


## Tips
- To render depth (e.g. `get_snapshot()`), it's recommended to set `LIBGL_ALWAYS_SOFTWARE` to `1`. In RTX4060 + Ubuntu20.04-WSL, Hardware rendering leads to abnormal depth.
- Codes below """ visualization """ or """ \<Tune\> xxx """ can be safely commented/uncommented to see how's the program going.
