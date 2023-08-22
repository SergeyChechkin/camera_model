/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>
#include <cmath>

template<typename T>
class ProjectionModel {
public:
    virtual Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) const = 0; 
    virtual Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) const = 0;
    virtual Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) const = 0;
    virtual Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) const = 0;  // elevation, azimuth
};

template<typename T>
class Perspective final : public ProjectionModel<T> {
public:
    static constexpr size_t param_size_ = 1;
public:
    Perspective(const T params[param_size_]) : f_(params[0]), inv_f_(T(1.0) / params[0]) {}
    Perspective(const T f) : f_(f), inv_f_(T(1.0) / f) {}
    Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) const override {
        if (abs(point[2]) < std::numeric_limits<T>::epsilon()) 
            return {T(0.0), T(0.0)};

        return {f_ * point[0] / point[2], f_ * point[1] / point[2]};
    } 

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) const override {
        return {inv_f_ * point[0], inv_f_ * point[1], T(1.0)};
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) const override {
        return ReProjectToUnitPlane(point).normalized();
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) const override { 
        throw std::logic_error("Unsupported method.");
        return {};
    }
private:
    T f_;
    T inv_f_;
};

template<typename T>
class Equidistant final : public ProjectionModel<T> {
public:
    static constexpr size_t param_size_ = 1;
public:
    Equidistant(const T params[param_size_]) : f_(params[0]), inv_f_(T(1.0) / params[0]) {}
    Equidistant(const T f) : f_(f), inv_f_(T(1.0) / f) {}
    Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) const override {
        using std::acos;
        
        const T r_xyz = point.norm();
        if(std::numeric_limits<T>::epsilon() > r_xyz) {
            return {T(0.0), T(0.0)};
        } else {
            const auto& xy = point.block(0, 0, 2, 1);
            const T r_xy = xy.norm();
            const T elevation = acos(point[2] / r_xyz);
            const Eigen::Vector2<T> azimuth_vec = r_xy > std::numeric_limits<T>::epsilon() ? xy / r_xy : Eigen::Vector2<T>(T(0.0), T(0.0));
            return f_ * elevation * azimuth_vec;
        }
    } 

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) const override {
        throw std::logic_error("Unsupported method.");
        return {};
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) const override {
        using std::atan2;
        using std::sin;
        using std::cos;
        
        const Eigen::Vector2<T> point_ = inv_f_ * point;
        const T elevation = point_.norm();
        if(elevation < std::numeric_limits<T>::epsilon()) {
            return {T(0.0), T(0.0), T(1.0)};
        }

        const T inv_norm = T(1) / elevation;
        const T sin_theta = sin(elevation);
        const T cos_azimuth = point_[0] * inv_norm;
        const T sin_azimuth = point_[1] * inv_norm;
        return {sin_theta * cos_azimuth, sin_theta * sin_azimuth, cos(elevation)};
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) const override { 
        using std::atan2;
        using std::sin;
        using std::cos;

        const T elevation = point.norm();
        if(elevation < std::numeric_limits<T>::epsilon()) {
            return {T(0.0), T(0.0)};
        }

        const T azimuth = atan2(point[1], point[0]);
        return {elevation, azimuth};
    }
private:
    T f_;
    T inv_f_;
};

template<typename T>
class FieldOfView final : public ProjectionModel<T> {
public:
    static constexpr size_t param_size_ = 2;
public:
    FieldOfView(const T params[param_size_]) 
    : f_(params[0])
    , inv_f_(T(1.0) / params[0])
    , w_(params[1])
    , inv_w_(T(1.0) / params[1])
    , two_tan_w_2_(2 * tan(params[1] / 2)) {
    }

    FieldOfView(const T f, const T w) 
    : f_(f)
    , inv_f_(T(1.0) / f)
    , w_(w)
    , inv_w_(T(1.0) / w)
    , two_tan_w_2_(2 * tan(w / 2)) {
    }

    Eigen::Vector2<T> Project(const Eigen::Vector3<T>& point) const override {
        const auto& xy = point.block(0, 0, 2, 1);
        const T ru = xy.norm();

        if(std::numeric_limits<T>::epsilon() > ru) {
            return {T(0.0), T(0.0)};
        } else {
            const T rd = atan2(ru * two_tan_w_2_, point[2]) / w_;
            return (f_ * rd / ru) * xy;
        }
    } 

    Eigen::Vector3<T> ReProjectToUnitPlane(const Eigen::Vector2<T>& point) const override {
        const auto m = inv_f_ * point;
        const T rd = m.norm();
        
        if(std::numeric_limits<T>::epsilon() > rd) {
            return {T(0.0), T(0.0), T(1.0)};
        } else {
            const T k = sin(rd * w_) / (rd * two_tan_w_2_);
            return {k * m[0], k * m[1], cos(rd * w_)};
        }
    }

    Eigen::Vector3<T> ReProjectToUnitSphere(const Eigen::Vector2<T>& point) const override {
        return ReProjectToUnitPlane(point).normalized();
    }

    Eigen::Vector2<T> ToSpherical(const Eigen::Vector2<T>& point) const override { 
        throw std::logic_error("Unsupported method.");
        return {};
    }
private:
    T f_;
    T inv_f_;
    T w_;
    T inv_w_;
    T two_tan_w_2_;
};
