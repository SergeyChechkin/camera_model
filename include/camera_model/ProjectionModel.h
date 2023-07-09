/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
#include <cmath>

template<typename T>
class ProjectionModel {
public:
    virtual Eigen::Vector2<T> Project(const  Eigen::Vector3<T>& point) = 0; 
    virtual Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) = 0;
    virtual Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) = 0;
    virtual Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) = 0;  // elevation, azimuth
};

template<typename T>
class PerspectivePM final : public ProjectionModel<T> {
public:
    Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) override {
        if (abs(point[2]) < std::numeric_limits<T>::epsilon()) 
            return {T(0), T(0)};

        return {point[0] / point[2], point[1] / point[2]};
    } 

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) override {
        return {point[0], point[1], T(1.0)};
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) override {
        return ReProjectToUnitPlane(point).normalized();
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) override { 
        throw std::logic_error("Unsupported method.");
        return {};
    }
};

template<typename T>
class EquidistantPM final : public ProjectionModel<T> {
public:
    Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) override {
        using std::acos;
        
        const T r_xyz = point.norm();
        if(std::numeric_limits<T>::epsilon() > r_xyz) {
            return {zero, zero};
        } else {
            const auto& xy = point.block(0, 0, 2, 1);
            const T r_xy = xy.norm();
            const T elevation = acos(point[2] / r_xyz);
            const Eigen::Vector2<T> azimuth_vec = r_xy > std::numeric_limits<T>::epsilon() ? xy / r_xy : Eigen::Vector2<T>(zero, zero);
            return elevation * azimuth_vec;
        }
    } 

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) override {
        throw std::logic_error("Unsupported method.");
        return {};
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) override {
        using std::atan2;
        using std::sin;
        using std::cos;
        
        const T elevation = point.norm();
        if(elevation < std::numeric_limits<T>::epsilon()) {
            return {zero, zero, one};
        }

        const T inv_norm = T(1) / elevation;
        const T sin_theta = sin(elevation);
        const T cos_azimuth = point[0] * inv_norm;
        const T sin_azimuth = point[1] * inv_norm;
        return {sin_theta * cos_azimuth, sin_theta * sin_azimuth, cos(elevation)};
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) override { 
        using std::atan2;
        using std::sin;
        using std::cos;

        const T elevation = point.norm();
        if(elevation < std::numeric_limits<T>::epsilon()) {
            return {zero, zero};
        }

        const T azimuth = atan2(point[1], point[0]);
        return {elevation, azimuth};
    }
private:
    static constexpr T one = T(1);
    static constexpr T zero = T(0);
};
