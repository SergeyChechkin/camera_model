# Pinhole camera model 
Pinhole camera model and camera calibration.
## Dependencies:
Ceres solver, OpenCV, my Utils
## Note:
Pixel center is used for coordinates conversion from discrete to continuous image space.
```c++ 
cv::Point src;
cv::Point2f dst(src.x + 0.5f, src.y + 0.5f);
```
## EuRoC dataset example:
![calib_image](https://github.com/SergeyChechkin/camera_model/assets/6116876/d8629659-01ba-4026-8632-e30d9993356d)

![calib_image_rect](https://github.com/SergeyChechkin/camera_model/assets/6116876/720d29f8-9590-4e05-a3da-dea39321a3aa)
