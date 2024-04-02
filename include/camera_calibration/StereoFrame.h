#pragma once

#include <opencv2/core.hpp>

class StereoFrame {
public:
    StereoFrame(const cv::Mat& left, const cv::Mat& right)
    : left_(left)
    , right_(right)
    , inv_depth_(left.rows, left.cols, CV_32F, cv::Scalar(1.0f)) 
    , inv_depth_sigma_(left.rows, left.cols, CV_32F, cv::Scalar(std::numeric_limits<float>::max()))
    {
        
    }  

    cv::Mat Display() const 
    {
        const int width = left_.cols;
        const int height = left_.rows;
        cv::Mat result(2 * height, 2 * width, CV_8U, cv::Scalar(0));

        cv::Mat top_left(result, cv::Range(0, height), cv::Range(0, width));
        left_.copyTo(top_left);

        cv::Mat top_right(result, cv::Range(0, height), cv::Range(width, 2 * width));
        right_.copyTo(top_right);

        cv::Mat inv_depth_8;
        inv_depth_.convertTo(inv_depth_8, CV_8U, 255.0 / 10);    
        cv::Mat bottom_left(result, cv::Range(height, 2 * height), cv::Range(0, width));
        inv_depth_8.copyTo(bottom_left);
        
        cv::Mat depth_sigma_8;
        inv_depth_sigma_.convertTo(depth_sigma_8, CV_8U, 255.0 / 4);
        cv::Mat bottom_right(result, cv::Range(height, 2 * height), cv::Range(width, 2 * width));
        depth_sigma_8.copyTo(bottom_right);

        return result;
    }  
public:
    cv::Mat left_;
    cv::Mat right_;
    cv::Mat inv_depth_;
    cv::Mat inv_depth_sigma_;
};

class RectifedStereoFrame : public StereoFrame {
public:
    RectifedStereoFrame(const cv::Mat& left, const cv::Mat& right, double stereo_base) 
    : StereoFrame(left, right)
    , stereo_base_(stereo_base)
    {

    } 
public:
    double stereo_base_;
};