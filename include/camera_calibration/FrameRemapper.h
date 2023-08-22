#pragma once

#include "camera_model/GeometricCameraModel.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

template<typename ScalarT>
class FrameRemapper {
public:
    FrameRemapper(const GeometricCameraModel<ScalarT>& src_model, const GeometricCameraModel<ScalarT>& dst_model, const Eigen::Matrix3d& rot_mat) 
    : src_model_(src_model)
    , dst_model_(dst_model)
    {
        GenerateRemap(rot_mat);
    }

    cv::Mat Remap(const cv::Mat& src_img) const {
        cv::Mat dst_img;
        cv::remap(src_img, dst_img, map_x_, map_y_, cv::InterpolationFlags::INTER_CUBIC);
        return dst_img;
    }
private:
    void GenerateRemap(const Eigen::Matrix<double, 3, 3>& rot_mat) 
    {
        const auto dst_image_size = dst_model_.GetImageSize();
        map_x_ = cv::Mat(dst_image_size[1], dst_image_size[0], CV_32F);
        map_y_ = cv::Mat(dst_image_size[1], dst_image_size[0], CV_32F);
        
        cv::parallel_for_(cv::Range(0, dst_image_size[1]), [&](const cv::Range & range){
            for(int v = range.start; v < range.end; ++v) {
                for(int u = 0; u < dst_image_size[0]; ++u) {
                    Eigen::Vector2d src(u + 0.5, v + 0.5);
                    Eigen::Vector3d src_3d = rot_mat * dst_model_.ReProjectToUnitSphere(src);
                    Eigen::Vector2d dst = src_model_.Project(src_3d);
                    map_x_.at<float>(v, u) = dst.x();
                    map_y_.at<float>(v, u) = dst.y();
                }
            }
        });     
    }
private:
    const GeometricCameraModel<ScalarT>& src_model_;
    const GeometricCameraModel<ScalarT>& dst_model_;
    cv::Mat map_x_; 
    cv::Mat map_y_;

};