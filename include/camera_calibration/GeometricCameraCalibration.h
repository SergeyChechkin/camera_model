#pragma once

#include "camera_model/GeometricCameraModel.h"

#include "utils/CeresUtils.h"
#include <ceres/ceres.h>
#include <glog/logging.h>

class CameraCalibrationSolver {
public:
    using Point3D = Eigen::Vector3d;
    using Point2D = Eigen::Vector2d;
public:
    void AddFrame(
        int id,
        const std::vector<Point3D>& object_points, 
        const std::vector<Point2D>& image_points,
        const std::vector<double>& weights);
    
    template<typename CameraModelGeneratorT>
    void Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params, 
        Eigen::Matrix<double, CameraModelGeneratorT::param_size_, CameraModelGeneratorT::param_size_>& info_mat);
private:
    void NormalizeSpatialDensity();
public:
    struct Frame {
        int id_;
        std::vector<Point3D> object_points_;
        std::vector<Point2D> image_points_;
        std::vector<double> weights_;
        Eigen::Vector<double, 6> pose_;
    };

    std::vector<Frame> frames_;
    static constexpr double cell_size_ = 20;
private:
    template<typename CameraModelGeneratorT>
    class CalibrationCF {
    public:
        template<typename T>
        bool operator()(const T pose[6], const T camera_params[CameraModelGeneratorT::param_size_], T residuals[2]) const {
            const Eigen::Vector3<T> gloabal_point = object_point_.cast<T>();        
            const Eigen::Vector3<T> camera_point = TransformPoint(pose, gloabal_point.data());
            const auto cm = CameraModelGeneratorT::Create(camera_params);
            const Eigen::Vector2<T> projection = cm.Project(camera_point);
            
            const Eigen::Vector2<T> error = T(weight_) * (projection - image_point_.cast<T>());
            residuals[0] = error[0];
            residuals[1] = error[1];

            return true;
        }

        static ceres::CostFunction* Create(
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point,
            double weight = 1.0) {
                return new ceres::AutoDiffCostFunction<CalibrationCF<CameraModelGeneratorT>, 2, 6, CameraModelGeneratorT::param_size_>(
                    new CalibrationCF<CameraModelGeneratorT>(object_point, image_point, weight));
            }

        CalibrationCF(
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point,
            double weight = 1.0) 
            : object_point_(object_point)
            , image_point_(image_point)
            , weight_(weight){}

    private:
        Eigen::Vector3d object_point_;
        Eigen::Vector2d image_point_;
        double weight_;
    };
};

template<typename CameraModelGeneratorT>
void CameraCalibrationSolver::Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params, 
        Eigen::Matrix<double, CameraModelGeneratorT::param_size_, CameraModelGeneratorT::param_size_>& info_mat)
{
    NormalizeSpatialDensity();

    for(auto& frame : frames_) {
        // initial pose estimation for all frame
        ceres::Problem problem;

        for(size_t i = 0; i < frame.object_points_.size(); ++i) {
            const auto& object_point = frame.object_points_[i];
            const auto& image_point = frame.image_points_[i];
            const double weight = frame.weights_[i];
            ceres::CostFunction* cost_function = CalibrationCF<CameraModelGeneratorT>::Create(object_point, image_point, weight);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(2.0), frame.pose_.data(), camera_params.data());
        }

        problem.SetParameterBlockConstant(camera_params.data());
        Optimize(problem, false, 10, 1e-3);
    }

    ceres::Problem problem;

    for(auto& frame : frames_) {
        for(size_t i = 0; i < frame.object_points_.size(); ++i) {
            const auto& object_point = frame.object_points_[i];
            const auto& image_point = frame.image_points_[i];
            const double weight = frame.weights_[i];

            ceres::CostFunction* cost_function = CalibrationCF<CameraModelGeneratorT>::Create(object_point, image_point, weight);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(2.0), frame.pose_.data(), camera_params.data());
        }
    }

    Optimize(problem, true, 1000, 1e-9);  

    /// Extracting intrinsic information matric from final solution.
    info_mat.setZero();
    static constexpr size_t jet_size = CameraModelGeneratorT::param_size_ + 6;
    using JetT = ceres::Jet<double, jet_size>;
    Eigen::Vector<JetT, CameraModelGeneratorT::param_size_> params_j;

    for(size_t i = 0; i < CameraModelGeneratorT::param_size_; ++i) {
        params_j[i] = JetT(camera_params[i], i); 
    }

    Eigen::Vector<JetT, 6> pose_j;
    for(auto& frame : frames_) {
        for(size_t i = 0; i < 6; ++i) {
            pose_j[i] = JetT(frame.pose_[i], CameraModelGeneratorT::param_size_ + i); 
        }
    
        for(size_t i = 0; i < frame.object_points_.size(); ++i) {
            const auto& object_point = frame.object_points_[i];
            const auto& image_point = frame.image_points_[i];
            const double weight = frame.weights_[i];

            Eigen::Vector<JetT, 2> res_j;
            CalibrationCF<CameraModelGeneratorT> cf(object_point, image_point, weight);
            cf.operator()(pose_j.data(), params_j.data(), res_j.data());
            
            Eigen::Matrix<double, 2, CameraModelGeneratorT::param_size_> J;
            J.row(0) = res_j[0].v.block(0, 0, CameraModelGeneratorT::param_size_, 1);
            J.row(1) = res_j[1].v.block(0, 0, CameraModelGeneratorT::param_size_, 1);
            info_mat += J.transpose() * J;
        }
    }
}