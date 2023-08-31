/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#pragma once

#include <Eigen/Core>

/// @brief Pinhole camera model. 
/// Perspective projection. Radial, decentering and Thin prism distortions.
/// 10 camera model parameters total.
/// @tparam T - scalar type 
template<typename T>
class PinholeCameraModel {
public:
    using Vector2D = Eigen::Vector2<T>;
    using Vector3D = Eigen::Vector3<T>;

    PinholeCameraModel(const T params[10]) {
        std::copy(params, params + 10, params_.data()); 
    }

    /// @brief Convert 3D point in camera frame to image frame
    /// @param point - 3D point
    /// @return - image point
    Vector2D Project(const Vector3D& point) const {
        if (abs(point[2]) < std::numeric_limits<T>::epsilon()) 
            return Vector2D(params_[2], params_[3]);

        Vector2D prjct(params_[0] * point[0] / point[2], params_[1] * point[1] / point[2]);
        return prjct + Distort(prjct) + Vector2D(params_[2], params_[3]);
    } 

    /// @brief Convert image point to 3D point in camera frame 
    /// @param point - image point
    /// @return - 3D point
    Vector3D ReProject(const Vector2D& point) const {
        Vector2D dist_prjct = point - Vector2D(params_[2], params_[3]);
        Vector2D undist_prjct = dist_prjct - Undistort(dist_prjct);
        return Vector3D(undist_prjct[0] / params_[0], undist_prjct[1] / params_[1], T(1));
    } 

    /// @brief Convert point from image frame to unit plane frame 
    /// @param point - image point 
    /// @return - unit plane point 
    Vector2D ImageToUnitPlane(const Vector2D& point) const {
        Vector2D dist_prjct = point - Vector2D(params_[2], params_[3]);
        Vector2D undist_prjct = dist_prjct - Undistort(dist_prjct);
        return Vector2D(undist_prjct[0] / params_[0], undist_prjct[1] / params_[1]);
    } 

    Vector2D Project(const T point[3]) const {
        return Project(Vector3D(point));
    }

    Vector3D ReProject(const T point[2]) const {
        return ReProject(Vector2D(point));
    } 

    const std::array<T, 10>& Params()const {return params_;}
private:
    Vector2D Distort(const Vector2D& r) const {
        const T rx = r.x();
        const T ry = r.y();
        const T rx_2 = rx * rx;
        const T ry_2 = ry * ry;
        const T rx_ry_2 = T(2) * rx * ry;
        const T r_2 = rx_2 + ry_2;
        
        // Radial distortion: 
        const Vector2D rad_dist = r * (params_[4] * r_2 + params_[5] * r_2 * r_2);
        // Decentering distortion:
        const Vector2D dsnt_dist(params_[6] * (T(3) * rx_2 + ry_2) + params_[7] * rx_ry_2, params_[7] * (T(3) * ry_2 + rx_2) + params_[6] * rx_ry_2);
        // Thin prism distortion:
        const Vector2D th_pr_dist(params_[8] * r_2, params_[9] * r_2);
        return rad_dist + dsnt_dist + th_pr_dist;
    }

    Vector2D Undistort(const Vector2D& dr) const {
        // Iterative solution without J computation (J close to 1).
        Vector2D r = dr;
        Vector2D distortion = Distort(r);

        for(size_t i = 0; i < max_iterations_; ++i) {
            Vector2D error = dr - (r + distortion);
            r = dr - distortion;
            if(error.squaredNorm() < std::numeric_limits<T>::epsilon()) {
                break;
            }
            distortion = Distort(r);
        }

        return distortion;
    }
    
    std::array<T, 10> params_;
    size_t max_iterations_ = 20;
};
