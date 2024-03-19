#pragma once

#include "CalibrationPattern.h" 

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>

#include <iostream>

template <typename T>
class FlatCheckerboardDetector_OpenCV {
public:
    using PatternType = FlatCheckerboardPattern<T>;
public:
    static constexpr int feature_patch_size_ = 11;
public:
    static bool DetectPattern(
        const cv::Mat& gray_image,
        const PatternType& pattern,
        std::vector<Eigen::Vector3<T>>& pattern_points, 
        std::vector<Eigen::Vector2<T>>& image_points) 
    {        
        pattern_points = pattern.GetSaddleCorners();
        image_points.clear();

        std::vector<cv::Point2f> image_corners;
        const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
        const cv::Size cv_pattern_size(pattern.pattern_size_[0] - 1, pattern.pattern_size_[1] - 1);

        if(!cv::findChessboardCorners(gray_image, cv_pattern_size, image_corners, flags)) {
            return false;
        }

        const cv::TermCriteria termCrit(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray_image, image_corners, cv::Size(feature_patch_size_, feature_patch_size_), cv::Size(-1, -1), termCrit);
        
        image_points.reserve(image_corners.size());

        for(size_t i = 0; i < image_corners.size(); ++i) {
            // TODO: implement convention for pixel position convertion 
            image_points.emplace_back(image_corners[i].x, image_corners[i].y);
            //image_points.emplace_back(image_corners[i].x + 0.5f, image_corners[i].y + 0.5f);    // + 0.5f - center pixel convention
        }

        return true;
    }
        
    // Draw pattern features on the image
    static void DrawDetectedPattern(
        cv::Mat& color_image, 
        const PatternType& pattern, 
        const std::vector<Eigen::Vector2<T>>& image_points)
    {
        std::vector<cv::Point2f> cv_image_corners;
        for(const auto& point : image_points) {
            cv_image_corners.emplace_back(point[0], point[1]);
        }

        const cv::Size cv_pattern_size(pattern.pattern_size_[0] - 1, pattern.pattern_size_[1] - 1);
        cv::drawChessboardCorners(color_image, cv_pattern_size, cv_image_corners, true);
    }
};