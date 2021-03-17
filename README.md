## Point cloud interpolater

A point interpolation library for LiDAR using images.

### Features

- Supports Linear, IP-Basic, Markov Random Field, Pixel weighted average strategy, Guided Filter, and Original method.
- Interpolate point cloud (pcd) upto same resolution as the image (png)

### Requirements

- C++ 17
- PCL 1.8
- OpenCV 3.4.2

### How to use

1. Create data folder

- Format

```
folder_name
├──xxx.png
├──xxx.pcd
├──yyy.png
├──yyy.pcd
├──...
```

You should specify same name for the image and point cloud data from the same frame.

Example) xxx.png and xxx.pcd.

PNG images are only supported.

2. Build this project

In this project,

```
$ mkdir build
$ cd build
$ make
```

3. Run

```
$ ./Interpolater <folder_path> <calibration_id> <method_name>
```

If you want to output the result to file,

```
$ ./Interpolater <folder_path> > <calibration_id> <method_name> > <output_path>
```

ex)

```
$ ./Interpolater ~/data miyanosawa_20200303 original
```
