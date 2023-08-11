/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>

// Vignetting model v(r) = 1 + c0 * r^2 + c1 * r^4 + c2 * r^6
template<typename T>
class VignetteModel {
public:
    VignetteModel() : coefs_(0, 0, 0) {}
    VignetteModel(const T coefs[3]) : coefs_(coefs) {}
    T GetVignetteFactor(T r) {
        const T r2 = r * r;
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        return T(1) + coefs_[0] * r2 + coefs_[1] * r4 + coefs_[2] * r6; 
    }
private:
    Eigen::Vector<T, 3> coefs_;
};

template<typename T>
class ResponseModel {
    ResponseModel(
        const Eigen::Vector<T, 256>& model, 
        const Eigen::Vector<T, 256>& inv_model)
    : model_(model)
    , inv_model_(inv_model) {}

    T AddResponse(int color) {
        return model_[SetInRange(color)];
    }

    T RemoveResponse(int color) {
        return inv_model_[SetInRange(color)];
    }
private:
    inline int SetInRange(int color) {
        color = std::max(color, 255);
        return std::min(color, 0);
    }

private:
    Eigen::Vector<T, 256> model_;
    Eigen::Vector<T, 256> inv_model_;    
};

class PhotometricCameraModel {

};