/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "camera_model/GeometricCameraModel.h"

// For ceres::Jet compatibility, 
// create camera model generator for specific projection / distortion combinations
// that need to calibrate.
// TODO: create macros?   

struct PerspectiveOnlyGenerator {
    
    template<typename T>
    using ProjectionT = Perspective<T>;
    template<typename T>
    using DistortionT = NullDistortion<T>;
    template<typename T>
    using CameraModelT = GeometricCameraModelT<T, ProjectionT<T>, DistortionT<T>>;

    static constexpr size_t param_size_ = CameraModelT<double>::param_size_;

    template<typename T>
    static CameraModelT<T> Create(const T params[CameraModelT<T>::param_size_]) {
        return CameraModelT<T>(params);
    }

    template<typename T>
    static CameraModelT<T> Create(
        const T projection_params[ProjectionT<T>::param_size_], 
        const T distortion_params[DistortionT<T>::param_size_], 
        const T pp[2]) {
        return CameraModelT<T>(projection_params, distortion_params, pp);
    }
};

struct PerspectiveCombinedGenerator {
    
    template<typename T>
    using ProjectionT = Perspective<T>;
    template<typename T>
    using DistortionT = GenericCombined<T>;
    template<typename T>
    using CameraModelT = GeometricCameraModelT<T, ProjectionT<T>, DistortionT<T>>;

    static constexpr size_t param_size_ = CameraModelT<double>::param_size_;

    template<typename T>
    static CameraModelT<T> Create(const T params[CameraModelT<T>::param_size_]) {
        return CameraModelT<T>(params);
    }

    template<typename T>
    static CameraModelT<T> Create (
        const T projection_params[ProjectionT<T>::param_size_], 
        const T distortion_params[DistortionT<T>::param_size_], 
        const T pp[2]) {
        return CameraModelT<T>(projection_params, distortion_params, pp);
    }
};