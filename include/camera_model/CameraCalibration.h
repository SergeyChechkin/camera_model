#pragma once

#include "CameraModel.h"
#include <opencv2/core.hpp>
#include <Eigen/Core>

class PinholeCameraCalibration {
public:
    PinholeCameraCalibration(cv::Size image_size, cv::Size pattern_size, cv::Size patch_size = {11, 11});
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
    //cv::Size dst_image_size_; 
};