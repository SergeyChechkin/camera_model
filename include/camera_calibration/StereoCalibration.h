#pragma once

#include <camera_calibration/CalibrationPattern.h>
#include <camera_calibration/GeometricCameraCalibration.h>
#include <camera_calibration/StereoCameraCalibration.h>

#include <opencv2/opencv.hpp>

class StereoCalibration {
private:
    FlatCheckerboardPattern<double> pattern_;
    StereoCameraCalibrationSolver solver_stereo_;
    CameraCalibrationSolver solver_l_;
    CameraCalibrationSolver solver_r_;
public:
    StereoCalibration(const FlatCheckerboardPattern<double>& pattern);
    
    cv::Mat DetectPattern(
        const cv::Mat& gray_image_l, 
        const cv::Mat& gray_image_r, 
        std::vector<Eigen::Vector3d>& pattern_points, 
        std::vector<Eigen::Vector2d>& image_points_l,
        std::vector<Eigen::Vector2d>& image_points_r);

    cv::Mat AddFrame(int id, const cv::Mat& gray_image_l, const cv::Mat& gray_image_r, bool visualize = true); 
    
    // Compute intrinsic parameters of both cameras 
    // and transformation between left and right camera.
    template<typename CameraModelGeneratorT>
    void Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l, 
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
        Eigen::Vector<double, 6>& pose_rl) 
    {
        solver_stereo_.Calibrate<CameraModelGeneratorT>(camera_params_l, camera_params_r, pose_rl);
    }
};