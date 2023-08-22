/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "camera_model/PinholeCameraModel.h"
#include <opencv2/core.hpp>
#include <Eigen/Core>

#include <ceres/rotation.h>

template<typename T>
Eigen::Vector3<T> GlobalToCamera(const T src[3], const T pose[6]) {
    Eigen::Vector3<T> result;
    ceres::AngleAxisRotatePoint(pose, src, result.data());
    return result + Eigen::Vector3<T>(pose + 3);
}

template<typename T, typename CameraT, size_t Nm>
Eigen::Vector2<T> GlobalToImage(const T src[3], const T pose[6], const T camera_params[Nm]) {
    Eigen::Vector3<T> camera_point = GlobalToCamera(src, pose);
    CameraT cm(camera_params);
    return cm.Project(camera_point);
}

class PinholeCameraCalibration {
public:
    PinholeCameraCalibration(cv::Size image_size, cv::Size pattern_size, cv::Size patch_size = {11, 11});
    void AddFrame(const std::vector<cv::Point2f>& image_points, const std::vector<cv::Point3f>& object_points);
    bool AddFrame(const cv::Mat& frame, cv::Mat* result = nullptr);
    void Calibrate(std::array<double, 10>& camera_params);
private:
    void CalculateWeight();
private:
    cv::Size image_size_;
    cv::Size pattern_size_;
    cv::Size patch_size_;
    int flags_;

    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<std::vector<cv::Point3f>> object_points_;
    std::vector<std::vector<float>> points_weights_; 
    std::vector<std::array<double, 6>> poses_;
};

class PinholeCameraRemap {
public: 
    PinholeCameraRemap(
        const std::array<double, 10>& src_camera_params, 
        const std::array<double, 10>& dst_camera_params);
    void GenerateDistortionRemap(
        cv::Size dst_image_size,
        const Eigen::Matrix3d& rot,
        cv::Mat& dx, 
        cv::Mat& dy);
    Eigen::Vector2d RemapPoint(const Eigen::Vector2d& src);
private:
    PinholeCameraModel<double> src_cm_;
    PinholeCameraModel<double> dst_cm_;
};