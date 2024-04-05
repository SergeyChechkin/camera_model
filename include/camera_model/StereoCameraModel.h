// BSD 3-Clause License
/// Copyright (c) 2022, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "camera_model/GeometricCameraModel.h"
#include <Eigen/Geometry>

template<typename CameraModelT>
struct StereoCameraModel {
    StereoCameraModel() 
    {

    }

    StereoCameraModel(
        const CameraModelT& cm_l,
        const CameraModelT& cm_r,
        const Eigen::Isometry3d& pose_rl) 
    : cm_l_(cm_l)
    , cm_r_(cm_r)
    , pose_rl_(pose_rl)
    {

    }

    CameraModelT cm_l_;
    CameraModelT cm_r_;
    Eigen::Isometry3d pose_rl_;
};