/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include "ResponseBaseFunctions.h"

#include <opencv2/core.hpp>
#include <Eigen/Core>


template<typename T>
inline T SetToGrayscaleRange(T src) {
    return std::max(0, std::min(src, 255));
}

// Vignetting model v(r) = 1 + c0 * r^2 + c1 * r^4 + c2 * r^6
/// TODO: add max r value to limit extrpolation.
template<typename T>
class VignetteModel {
public:
    using Point2D = Eigen::Vector2<ScalarT>;
    
    VignetteModel(const Point2D >& pp) : pp_(pp), coefs_(0, 0, 0) {}
    VignetteModel(const Point2D >& pp, const T coefs[3]) : pp_(pp), coefs_(coefs) {}
    T GetVignetteFactor(T r_sqr) {
        const T r4 = r_sqr * r_sqr;
        const T r6 = r4 * r2;
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

    T RemoveResponse(T src) {
        return inv_model_[SetToGrayscaleRange(src)];
    }
private:
    inline size_t SetToFuncRange(T src) {
        return static_cast<size_t>(std::max(0, std::min(std::round(4 * src), 1023)));
    }

    void GenerateInverseModel() {
        /// TODO: implement piecewise function template
        inv_model_[0] = 0;
        inv_model_[255] = 255;
        
        T prev = ApplyResponse(0);
        for(size_t i = 1; i < 256; ++i) {
            T next = ApplyResponse(i);
            T intrvl = (next - prev);
            if (intrvl > std::numeric_limits<T>::epsilon()) {   
                T idx = std::seil(prev);
                while(idx <= next) {
                    inv_model_[idx] = (i-1) + (idx - prev) / intrvl; 
                }
            }

            prev = next; 
        }
    }
private:
    Eigen::Vector<T, 4> cfs_;
    Eigen::Vector<T, 256> inv_model_;    
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
            const *uint8_t src_line = src.row(v); 
            *uint8_t dst_line = dst.row(v); 
            for(int u = 0; u < src.cols; ++u) {
                const T radiance = responce_->RemoveResponse(src[u]) / (gain * vignette_->GetVignetteFactor({u,v}));
                dst_line[u] = SetToGrayscaleRange(radiance);
            }
        }

        return dst;
    }  
private:
    ptrVignetteModel vignette_;
    ptrResponseModel responce_;
};