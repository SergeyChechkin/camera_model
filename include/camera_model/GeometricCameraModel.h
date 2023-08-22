/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "camera_model/ProjectionModel.h"
#include "camera_model/DistortionModel.h"

#include <opencv2/core.hpp>
#include <Eigen/Core>

template<typename ScalarT>
class GeometricCameraModel {
public:
    using Point3D = Eigen::Vector3<ScalarT>;
    using Point2D = Eigen::Vector2<ScalarT>;
    using SphericalVector = Eigen::Vector2<ScalarT>; // elevation, azimuth
    using ProjectionT = ProjectionModel<ScalarT>;
    using DistortionT = DistortionModel<ScalarT>;
    using ptrProjectionT = std::shared_ptr<ProjectionT>;
    using ptrDistortionT = std::shared_ptr<DistortionT>;
public:
    // Constructiors
    GeometricCameraModel(
        const ptrProjectionT projection, 
        const ptrDistortionT distortion, 
        const Point2D& pp, 
        const Eigen::Vector2i& image_size) 
    : projection_(projection)
    , distortion_(distortion)
    , pp_(pp)
    , image_size_(image_size)
    {}
    
    GeometricCameraModel(
        const ptrProjectionT projection, 
        const ptrDistortionT distortion, 
        const Point2D& pp, 
        const cv::Mat& mask) 
    : projection_(projection)
    , distortion_(distortion)
    , pp_(pp)
    , image_size_(mask.cols, mask.rows)
    , mask_(mask)
    {}

public:
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
    
    SphericalVector ConvertToSpherical(const Point2D& point) const
    {
        return projection_->ToSpherical(Undistort(point));
    } 

    SphericalVector ConvertToSpherical(const Point3D& point) const
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
    
    inline Eigen::Vector2i GetImageSize() const {return image_size_;}
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
    Point2D pp_;                        // principal point
    Eigen::Vector2i image_size_;        // image size
    cv::Mat mask_;                      // mask image, defined valid area of the image
};


template<typename ScalarT, typename ProjectionT, typename DistortionT>
class GeometricCameraModelT {
public:
    using Point3D = Eigen::Vector3<ScalarT>;
    using Point2D = Eigen::Vector2<ScalarT>;
    using SphericalVector = Eigen::Vector2<ScalarT>; // elevation, azimuth
public:
    static size_t GetProjectionParamsSize() {return ProjectionT::param_size_;}
    static size_t GetDistortionParamsSize() {return DistortionT::param_size_;}
    static constexpr size_t param_size_ = ProjectionT::param_size_ + DistortionT::param_size_ + 2;
public:
    // Constructiors
    GeometricCameraModelT(
        const ScalarT projection_params[ProjectionT::param_size_], 
        const ScalarT distortion_params[DistortionT::param_size_], 
        const ScalarT pp[2])
    : projection_(projection_params)
    , distortion_(distortion_params)
    , pp_(Point2D(pp))
    , image_size_(0, 0)
    {}

    GeometricCameraModelT(
        const ScalarT params[param_size_])
    : projection_(params)
    , distortion_(params + ProjectionT::param_size_ + 2)
    , pp_(Point2D(params + ProjectionT::param_size_))
    , image_size_(0, 0)
    {}

    GeometricCameraModelT(
        const ScalarT projection_params[ProjectionT::param_size_], 
        const ScalarT distortion_params[DistortionT::param_size_], 
        const Point2D pp, 
        const Eigen::Vector2i& image_size) 
    : projection_(projection_params)
    , distortion_(distortion_params)
    , pp_(pp)
    , image_size_(image_size)
    {}

    GeometricCameraModelT(
        const ScalarT projection_params[ProjectionT::param_size_], 
        const ScalarT distortion_params[DistortionT::param_size_], 
        const Point2D pp, 
        const cv::Mat& mask) 
    : projection_(projection_params)
    , distortion_(distortion_params)
    , pp_(pp)
    , image_size_(mask.cols, mask.rows)
    , mask_(mask)
    {}
public:
    Point2D Project(const Point3D& point) const
    {
        const auto prjct = projection_.Project(point);
        return prjct + distortion_.Distort(prjct) + pp_;
    }

    Point3D ReProjectToUnitPlane(const Point2D& point) const
    {
        return projection_.ReProjectToUnitPlane(Undistort(point));
    }

    Point3D ReProjectToUnitSphere(const Point2D& point) const
    {
        return projection_.ReProjectToUnitSphere(Undistort(point));
    }
    
    SphericalVector ConvertToSpherical(const Point2D& point) const
    {
        return projection_.ToSpherical(Undistort(point));
    } 

    SphericalVector ConvertToSpherical(const Point3D& point) const
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
    
    inline Eigen::Vector2i GetImageSize() const {return image_size_;}
private:
    inline Point2D Undistort(const Point2D& point) const 
    {
        const auto dist_prjct = point - pp_;
        return dist_prjct - distortion_.Undistort(dist_prjct);
    }

    inline bool CheckImageRect(const Point2D& point) const 
    {
        return !(point[0] < 0 || point[1] < 0 || point[0] >= image_size_[0] || point[1] >= image_size_[1]);
    }
private:
    ProjectionT projection_;            // projection model
    DistortionT distortion_;            // distortion model
    Point2D pp_;                        // principal point
    Eigen::Vector2i image_size_;        // image size
    cv::Mat mask_;                      // mask image, defined valid area of the image
};