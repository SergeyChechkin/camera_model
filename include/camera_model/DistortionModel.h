/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
#include <cmath>

template<typename T>
class DistortionModel {
public:
    virtual Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) const = 0; 
    virtual Eigen::Vector2<T> Undistort(const Eigen::Vector2<T>& dr) const {
        // Iterative solution without J computation (J close to I).
        const size_t max_iterations = 20;   // usually less than 8 
        Eigen::Vector2<T> r = dr;
        Eigen::Vector2<T> distortion = Distort(r);

        for(size_t i = 0; i < max_iterations; ++i) {
            const Eigen::Vector2<T> error = dr - (r + distortion);
            r = dr - distortion;
            if(error.squaredNorm() < std::numeric_limits<T>::epsilon()) {
                break;
            }
            distortion = Distort(r);
        }

        return distortion;
    } 
};

template<typename T, size_t Nm>
class RadialPolynomial final : public DistortionModel<T> {
public:
    RadialPolynomial(const T params[Nm]) {
        std::copy(params, params + Nm, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) const override {
        T r_2 = r.squaredNorm();
        
        T amplitude = T(0);
        T r_pow = T(1);
        for(auto param : params_) {
            r_pow *= r_2;
            amplitude += r_pow * param;
        }

        return amplitude * r;
    } 
private:
    std::array<T, Nm> params_;
};

template<typename T>
class Decentering final : public DistortionModel<T> {
public:
    Decentering(const T params[2]) {
        std::copy(params, params + 2, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) const override {
        const T rx = r[0];
        const T ry = r[1];
        const T rx_2 = rx * rx;
        const T ry_2 = ry * ry;
        const T rx_ry_2 = T(2) * rx * ry;
        
        return {params_[0] * (T(3) * rx_2 + ry_2) + params_[1] * rx_ry_2, params_[1] * (T(3) * ry_2 + rx_2) + params_[0] * rx_ry_2};
    } 
private:
    std::array<T, 2> params_;
};

template<typename T>
class ThinPrism final : public DistortionModel<T> {
public:
    ThinPrism(const T params[4]) {
        std::copy(params, params + 4, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) const override {
        T r_2 = r.squaredNorm();
        T r_4 = r_2 * r_2;
        return {params_[0] * r_2 + params_[2] * r_4, params_[1] * r_2 + params_[3] * r_4};
    } 
private:
    std::array<T, 4> params_;
};

template<typename T>
class GenericCombined final : public DistortionModel<T> {
public:
    GenericCombined(const T params[6]) {
        std::copy(params, params + 6, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) const override {
        const T rx = r.x();
        const T ry = r.y();
        const T rx_2 = rx * rx;
        const T ry_2 = ry * ry;
        const T rx_ry_2 = T(2) * rx * ry;
        const T r_2 = rx_2 + ry_2;
        
        // Radial distortion: 
        const Eigen::Vector2<T> rad_dist = r * (params_[0] * r_2 + params_[1] * r_2 * r_2);
        // Decentering distortion:
        const Eigen::Vector2<T> dsnt_dist(params_[2] * (T(3) * rx_2 + ry_2) + params_[3] * rx_ry_2, params_[3] * (T(3) * ry_2 + rx_2) + params_[2] * rx_ry_2);
        // Thin prism distortion:
        const Eigen::Vector2<T> th_pr_dist(params_[4] * r_2, params_[5] * r_2);
        
        return rad_dist + dsnt_dist + th_pr_dist;
    } 
private:
    std::array<T, 6> params_;
};
