#pragma once

#include "camera_calibration/CameraRemap.h"
#include <Eigen/Core>


template<typename RawCameraModelT, typename RectCameraModelT>
class StereoRectification {
public:
    StereoRectification(
        const RawCameraModelT& raw_cm_l,
        const RawCameraModelT& raw_cm_r,
        const Eigen::Isometry3d& pose_rl,
        const RectCameraModelT& rect_cm,
        cv::Size rect_image_size, 
        double plane_rot_angle = 0.0)
    {
        Eigen::Matrix3d rect_rot_l;
        Eigen::Matrix3d rect_rot_r;      
        RectRotations(pose_rl, rect_rot_l, rect_rot_r, plane_rot_angle);

        remap_l_.GenerateRemap(raw_cm_l, rect_cm, rect_image_size, rect_rot_l);
        remap_r_.GenerateRemap(raw_cm_r, rect_cm, rect_image_size, rect_rot_r);
    } 
        
    void Rectify(
        const cv::Mat& src_image_l, 
        const cv::Mat& src_image_r, 
        cv::Mat& rect_image_l, 
        cv::Mat& rect_image_r)
    {
        rect_image_l = remap_l_.Remap(src_image_l);
        rect_image_r = remap_r_.Remap(src_image_r);
    }
private:
    // Compute rectification rotations for both cameras 
    // to parallel optical axis and align X axis with epipolar plane.  
    // plane_rot_angle - epipolar plane rotation relative to Z-axis in left camera
    static void RectRotations(
        const Eigen::Isometry3d& pose_rl, 
        Eigen::Matrix3d& rect_rot_l, 
        Eigen::Matrix3d& rect_rot_r, 
        double plane_rot_angle)     
    {
        // Left camera pose is Idententy 
        // Right camera pose is pose_rl.
        // new X axis aligned with translation vector
        const Eigen::Vector3d new_axis_x = -(pose_rl.translation().normalized());

        // New Y axis is epipolar plane normal.  
        const Eigen::Vector3d old_axis_z_l = Eigen::Vector3d::UnitZ();
        Eigen::Vector3d new_axis_y = old_axis_z_l.cross(new_axis_x).normalized();
        new_axis_y = Eigen::AngleAxisd(plane_rot_angle, new_axis_x) * new_axis_y;

        rect_rot_l.col(0) = new_axis_x;
        rect_rot_l.col(1) = new_axis_y;
        rect_rot_l.col(2) = new_axis_x.cross(new_axis_y).normalized();

        rect_rot_r = rect_rot_l * pose_rl.rotation();
    } 
private:
    CameraRemap<RawCameraModelT, RectCameraModelT> remap_l_;
    CameraRemap<RawCameraModelT, RectCameraModelT> remap_r_;
};