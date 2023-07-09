/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
#include <cmath>

template<typename T>
class DistortionModel {
public:
    virtual Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) = 0; 
};

template<typename T, size_t Nm>
class RadialDistortionModel final : public DistortionModel<T> {
public:
    RadialDistortionModel(const T params[Nm]) {
        std::copy(params, params + Nm, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) override {
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
class DecenteringDistortionModel final : public DistortionModel<T> {
public:
    DecenteringDistortionModel(const T params[2]) {
        std::copy(params, params + 2, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) override {
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
class ThinPrismDistortionModel final : public DistortionModel<T> {
public:
    ThinPrismDistortionModel(const T params[2]) {
        std::copy(params, params + 2, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) override {
        T r_2 = r.squaredNorm();
        return {params_[0] * r_2, params_[1] * r_2};
    } 
private:
    std::array<T, 2> params_;
};

template<typename T, size_t Nm>
class CombinedDistortionModel final : public DistortionModel<T> {
public:
    CombinedDistortionModel(const T params[Nm]) {
        std::copy(params, params + Nm, params_.data()); 
    }

    Eigen::Vector2<T> Distort(const Eigen::Vector2<T>& r) override {
        const T rx = r.x();
        const T ry = r.y();
        const T rx_2 = rx * rx;
        const T ry_2 = ry * ry;
        const T rx_ry_2 = T(2) * rx * ry;
        const T r_2 = rx_2 + ry_2;
        
        // Decentering distortion:
        const Eigen::Vector2<T> dsnt_dist(params_[0] * (T(3) * rx_2 + ry_2) + params_[1] * rx_ry_2, params_[1] * (T(3) * ry_2 + rx_2) + params_[0] * rx_ry_2);
        // Thin prism distortion:
        const Eigen::Vector2<T> th_pr_dist(params_[2] * r_2, params_[3] * r_2);
        // Radial distortion: 
        T amplitude = T(0);
        T r_pow = T(1);
        for(size_t i = 4; i < Nm; ++i) {
            r_pow *= r_2;
            amplitude += r_pow * params_[i];
        }
        const Eigen::Vector2<T> rad_dist = r * amplitude;
        
        return rad_dist + dsnt_dist + th_pr_dist;
    } 
private:
    std::array<T, Nm> params_;
};
