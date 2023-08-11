/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>

template<typename ScalarT>
class GeometricCameraModel {
public:
    using Point3D = Eigen::Vector3<ScalarT>;
    using Point2D = Eigen::Vector2<ScalarT>;
    using SpericalVector = Eigen::Vector2<ScalarT>; // elevation, azimuth
    using ProjectionT = ProjectionModel<ScalarT>;
    using DistortionT = DistortionModel<ScalarT>;
    using ptrProjectionT = std::shared_ptr<ProjectionT>;
    using ptrDistortionT = std::shared_ptr<DistortionT>;
public:
    // Constructiors
    GeometricCameraModel(
        const ptrProjectionT projection, 
        const ptrDistortionT distortion, 
        const Eigen::Vector2<ScalarT>& pp, 
        const Eigen::Vector2i& image_size) 
    : projection_(projection)
    , distortion_(distortion)
    , pp_(pp)
    , image_size_(image_size)
    {}
    
    GeometricCameraModel(
        const ptrProjectionT projection, 
        const ptrDistortionT distortion, 
        const Eigen::Vector2<ScalarT> pp, 
        const cv::Mat& mask) 
    : projection_(projection)
    , distortion_(distortion)
    , pp_(pp)
    , image_size_(mask.cols, mask.rows)
    , mask_(mask)
    {}

public:
    // Projection methods

    Point2D Project(const Point3D& point) const
    {
        const auto prjct = projection_->Project(point);
        return prjct + distortion_->Distort(prjct) + pp_;
    }

    Point3D ReProjectToUnitPlane(const Point2D& point) const
    {
        return projection_->ReProjectToUnitPlane(Undistort(point));
    }

    Point3D ReProjectToUnitSphere(const Point2D& point) const
    {
        return projection_->ReProjectToUnitSphere(Undistort(point));
    }
    
    SpericalVector ConvertToSpherical(const Point2D& point) const
    {
        return projection_->ToSpherical(Undistort(point));
    } 

    SpericalVector ConvertToSpherical(const Point3D& point) const
    {
        const auto r = point.norm();
        if(r < std::numeric_limits<ScalarT>::epsilon()) {
            return {0, 0};
        } else {
            return {std::acos(point[2] / r), std::atan2(point[1], point[0])};
        }
    } 

    inline bool CheckImagePoint(const Point2D& point) const
    {
        return CheckImageRect(point) && (mask_.empty() || mask_.at<uint8_t>(std::floor(point[1]), std::floor(point[0])) > 0);
    }
private:
    inline Point2D Undistort(const Point2D& point) const 
    {
        const auto dist_prjct = point - pp_;
        return dist_prjct - distortion_->Undistort(dist_prjct);
    }

    inline bool CheckImageRect(const Point2D& point) const 
    {
        return !(point[0] < 0 || point[1] < 0 || point[0] >= image_size_[0] || point[1] >= image_size_[1]);
    }

private:
    const ptrProjectionT projection_;   // projection model
    const ptrDistortionT distortion_;   // distortion model
    Eigen::Vector2<ScalarT> pp_;        // principal point
    Eigen::Vector2i image_size_;        // image size
    cv::Mat mask_;                      // mask image, defined valid area of the image
};