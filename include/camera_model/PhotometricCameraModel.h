/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "ResponseBaseFunctions.h"

#include <utils/functions/PiecewiseFunction.h>

#include <opencv2/core.hpp>
#include <Eigen/Core>

#include <cmath>

template<typename T>
inline T SetToGrayscaleRange(T src) {
    return std::max<T>(0, std::min<T>(src, 255));
}

// Vignetting model v(r) = 1 + c0 * r^2 + c1 * r^4 + c2 * r^6
/// TODO: add max r value to limit extrpolation.
template<typename T>
class VignetteModel {
public:
    using Point2D = Eigen::Vector2<T>;
    
    VignetteModel(const Point2D& pp) : pp_(pp), coefs_(0, 0, 0) {}
    VignetteModel(const Point2D& pp, const T coefs[3]) : pp_(pp), coefs_(coefs) {}
    T GetVignetteFactor(T r_sqr) {
        const T r4 = r_sqr * r_sqr;
        const T r6 = r4 * r_sqr;
        return T(1) + coefs_[0] * r_sqr + coefs_[1] * r4 + coefs_[2] * r6; 
    }
    T GetVignetteFactor(const Point2D& image_point) {
        const T r_sqr = (image_point - pp_).squareNorm();
        return GetVignetteFactor(r_sqr);
    }
private:
    Point2D pp_;
    Eigen::Vector<T, 3> coefs_;
};

template<typename T>
class ResponseModel {
public:
    ResponseModel()
    : cfs_(0, 0, 0, 0) {
        GenerateInverseModel();
    }

    ResponseModel(const Eigen::Vector<T, 4>& cfs)
    : cfs_(cfs) {
        GenerateInverseModel();
    }

    /// @brief Applay reponse function   
    /// @param src - [0 .. 256) 
    /// @return [0 .. 256)
    T ApplyResponse(T src) {
        size_t src_func = SetToFuncRange(src);
        return SetToGrayscaleRange(256 * (f[src_func] + cfs_[0] * h_0[src_func] + cfs_[1] * h_1[src_func] + cfs_[2] * h_2[src_func] + cfs_[3] * h_3[src_func])); 
    }

    /// @brief Remove response 
    /// @param src - [0 .. 256) 
    /// @return [0 .. 256)
    T RemoveResponse(T src) {
        return inv_model_[SetToGrayscaleRange(src)];
    }
private:
    inline size_t SetToFuncRange(T src) {
        return static_cast<size_t>(std::max<T>(0, std::min<T>(std::round(4 * src), 1023)));
    }

    void GenerateInverseModel() {
        auto func = [this](const float& v) { return ApplyResponse(v);}; 
        inv_model_ = GeneratePiecewiseInverseFunction<T>(256, 0, 255, func);
    }
private:
    Eigen::Vector<T, 4> cfs_;
    std::vector<T> inv_model_;    
};

template<typename T>
class InverseResponseModel {
public:
    InverseResponseModel(const std::vector<T>& inv_model)
    : inv_model_(inv_model) {
    }

    /// @brief Applay reponse function   
    /// @param src - [0 .. 256) 
    /// @return [0 .. 256)
    T ApplyResponse(T src) {
        size_t src_func = SetToFuncRange(src);
        return SetToGrayscaleRange(256 * (f[src_func] + cfs_[0] * h_0[src_func] + cfs_[1] * h_1[src_func] + cfs_[2] * h_2[src_func] + cfs_[3] * h_3[src_func])); 
    }

    /// @brief Remove response 
    /// @param src - [0 .. 256) 
    /// @return [0 .. 256)
    T RemoveResponse(T src) {
        return inv_model_[SetToGrayscaleRange(src)];
    }
private:
    inline size_t SetToFuncRange(T src) {
        return static_cast<size_t>(std::max<T>(0, std::min<T>(std::round(4 * src), 1023)));
    }

    void GenerateInverseModel() {
        auto func = [this](const float& v) { return ApplyResponse(v);}; 
        inv_model_ = GeneratePiecewiseInverseFunction<T>(256, 0, 255, func);
    }
private:
    std::vector<T> inv_model_;    
};


template<typename T>
class PhotometricCameraModel {
public:
    using ptrVignetteModel = std::shared_ptr<VignetteModel<T>>;
    using ptrResponseModel = std::shared_ptr<ResponseModel<T>>;
    // image output intensity (raw pixel value) to radiance
    // assumption: grayscale image  
    cv::Mat CorrectFrame(const cv::Mat& src, T gain) {
        cv::Mat dst = src.clone();

        for(int v = 0; v < src.rows; ++v) {
            const uint8_t* src_line = src.ptr<uint8_t>(v);
            uint8_t* dst_line = dst.ptr<uint8_t>(v); 
            for(int u = 0; u < src.cols; ++u) {
                const T radiance = responce_->RemoveResponse(src_line[u]) / (gain * vignette_->GetVignetteFactor({u,v}));
                dst_line[u] = SetToGrayscaleRange(radiance);
            }
        }

        return dst;
    }  
private:
    ptrVignetteModel vignette_;
    ptrResponseModel responce_;
};