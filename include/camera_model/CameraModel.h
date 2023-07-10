/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "ProjectionModel.h"
#include "DistortionModel.h"

#include <memory>

template<typename T>
class CameraModel {
public:
    using Point3DT = Eigen::Vector3<T>;
    using Point2DT = Eigen::Vector2<T>;
public:
    virtual Point2DT Project(const Point3DT& point) const = 0;
    virtual Point3DT ReProjectToUnitPlane(const Point2DT& point) const = 0;
    virtual Point3DT ReProjectToUnitSphere(const Point2DT& point) const = 0;
    virtual Point2DT ToSpherical(const Point2DT& point) const = 0; // elevation, azimuth
}; 

template<typename T>
class UniversalCameraModel : public CameraModel<T> {
    using Point3DT = typename CameraModel<T>::Point3DT;
    using Point2DT = typename CameraModel<T>::Point2DT;
    using ProjectionT = ProjectionModel<T>;
    using DistortionT = DistortionModel<T>;
    using ptrProjectionT = std::shared_ptr<ProjectionT>;
    using ptrDistortionT = std::shared_ptr<DistortionT>;
public:
    UniversalCameraModel(const ptrProjectionT projection, const ptrDistortionT distortion, const Eigen::Vector2<T>& pp) 
    : projection_(projection)
    , distortion_(distortion)
    , pp_(pp)  
    { }

    Point2DT Project(const Eigen::Vector3<T>& point) const override {
        const auto prjct = projection_->Project(point);
        return prjct + distortion_->Distort(prjct) + pp_;
    }

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) const override {
        const Eigen::Vector2<T> dist_prjct = point - pp_;
        const Eigen::Vector2<T> undist_prjct = dist_prjct - distortion_->Undistort(dist_prjct);
        return projection_->ReProjectToUnitPlane(undist_prjct);
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) const override {
        const Eigen::Vector2<T> dist_prjct = point - pp_;
        const Eigen::Vector2<T> undist_prjct = dist_prjct - distortion_->Undistort(dist_prjct);
        return projection_->ReProjectToUnitSphere(undist_prjct);
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) const override {
        const Eigen::Vector2<T> dist_prjct = point - pp_;
        const Eigen::Vector2<T> undist_prjct = dist_prjct - distortion_->Undistort(dist_prjct);
        return projection_->ToSpherical(undist_prjct);
    } 
private:
    const ptrProjectionT projection_;
    const ptrDistortionT distortion_;
    Eigen::Vector2<T> pp_;  
}; 