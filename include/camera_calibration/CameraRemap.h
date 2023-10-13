/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

template<typename InCameraModelT, typename OutCameraModelT>
class CameraRemap {
public:
    void GenerateRemap(
        const InCameraModelT& in_cm, 
        const OutCameraModelT& out_cm,
        cv::Size out_image_size,
        const Eigen::Matrix3d& rot) {
            dx_ = cv::Mat(out_image_size.height, out_image_size.width, CV_32F);
            dy_ = cv::Mat(out_image_size.height, out_image_size.width, CV_32F);
    
            cv::parallel_for_(cv::Range(0, out_image_size.height), [&](const cv::Range & range){
                for(int v = range.start; v < range.end; ++v) {
                    for(int u = 0; u < out_image_size.width; ++u) {
                        Eigen::Vector2d src(u, v);
                        Eigen::Vector3d src_3d = rot * out_cm.ReProjectToUnitSphere(src);
                        Eigen::Vector2d dst = in_cm.Project(src_3d);
                        dx_.at<float>(v, u) = dst.x();
                        dy_.at<float>(v, u) = dst.y();
                    }
                }
            });     
        }
    cv::Mat Remap(const cv::Mat& img) {
        cv::Mat remap_img;
        cv::remap(img, remap_img, dx_, dy_, cv::InterpolationFlags::INTER_CUBIC);
        return remap_img;
    }

private:
    cv::Mat dx_; 
    cv::Mat dy_;
};