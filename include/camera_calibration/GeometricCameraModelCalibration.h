#pragma once

#include "camera_model/GeometricCameraModel.h"

#include "utils/CeresUtils.h"
#include <ceres/ceres.h>
#include <glog/logging.h>

class CameraCalibrationT {
public:
    using Point3D = Eigen::Vector3d;
    using Point2D = Eigen::Vector2d;
public:
    void AddFrame(
        const std::vector<Point3D>& object_points, 
        const std::vector<Point2D>& image_points,
        const std::vector<double>& weights);
    
    template<typename CameraModelGeneratorT>
    void Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params, 
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_info);
private:
    struct Frame {
        std::vector<Point3D> object_points_;
        std::vector<Point2D> image_points_;
        std::vector<double> weights_;
        Eigen::Vector<double, 6> pose_;
    };

    std::vector<Frame> frames_;
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
void CameraCalibrationT::Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params, 
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_info)
{
    for(auto& frame : frames_) {
        // initial pose estimation for all frame
        ceres::Problem problem;

        for(size_t i = 0; i < frame.object_points_.size(); ++i) {
            const auto& oblect_point = frame.object_points_[i];
            const auto& image_point = frame.image_points_[i];
            const double weight = frame.weights_[i];
            ceres::CostFunction* cost_function = CalibrationCF<CameraModelGeneratorT>::Create(oblect_point, image_point, weight);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(2.0), frame.pose_.data(), camera_params.data());
        }

        problem.SetParameterBlockConstant(camera_params.data());
        Optimize(problem, true, 10, 1e-3);
    }

    ceres::Problem problem;

    for(auto& frame : frames_) {
        for(size_t i = 0; i < frame.object_points_.size(); ++i) {
            const auto& oblect_point = frame.object_points_[i];
            const auto& image_point = frame.image_points_[i];
            const double weight = frame.weights_[i];

            ceres::CostFunction* cost_function = CalibrationCF<CameraModelGeneratorT>::Create(oblect_point, image_point, weight);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(2.0), frame.pose_.data(), camera_params.data());
        }
    }

    Optimize(problem, true, 1000, 1e-6);  

    /// TODO: Extract information matric from final solution.
}
