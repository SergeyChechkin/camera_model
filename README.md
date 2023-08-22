# Generalized camera model 
Generalized geometrical camera model is a  framework that allows combining any types of projection and distortion models.
# Pinhole camera model 
Pinhole camera model and camera calibration. Perspective projection model. Radial, Decentering and Thin prism distortion models. 10 model parameters total.
## Dependencies:
Ceres solver, OpenCV, my Utils
## Note:
Pixel center is used for coordinates conversion from discrete to continuous image space. What is differ from OpenCV convention. 
```c++ 
cv::Point src;
cv::Point2f dst(src.x + 0.5f, src.y + 0.5f);
```
## EuRoC dataset example:
![calib_image](https://github.com/SergeyChechkin/camera_model/assets/6116876/45d0a3a8-964c-4bd3-a585-0dcb098a0ca5)
![calib_image_rect](https://github.com/SergeyChechkin/camera_model/assets/6116876/1b995802-85b6-4e86-a1df-d22cfd775491)
