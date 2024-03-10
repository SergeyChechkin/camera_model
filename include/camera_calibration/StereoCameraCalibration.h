#pragma once

#include "camera_model/GeometricCameraModel.h"
#include "camera_calibration/GeometricCameraCalibration.h"

#include "utils/CeresUtils.h"
#include <ceres/ceres.h>
#include <glog/logging.h>

class StereoCameraCalibrationSolver {
public:
    using Point3D = Eigen::Vector3d;
    using Point2D = Eigen::Vector2d;
public:
    void AddFrame(
        int id,
        const std::vector<Point3D>& pattern_points, 
        const std::vector<Point2D>& image_points_l,
        const std::vector<Point2D>& image_points_r) {
            
            // TODO: replace veights with features covariance
            std::vector<double> weights(pattern_points.size(), 1.0);

            if (!image_points_l.empty()) {
                solver_l.AddFrame(id, pattern_points, image_points_l, weights);
            }

            if (!image_points_r.empty()) {
                solver_r.AddFrame(id, pattern_points, image_points_r, weights);
            }
            
            if (!image_points_l.empty() && !image_points_r.empty()) {
                CHECK_EQ(pattern_points.size(), image_points_l.size());
                CHECK_EQ(pattern_points.size(), image_points_r.size());
                ids_.push_back(id);
            }
        }   

    template<typename CameraModelGeneratorT>
    void Calibrate(
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
        Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
        Eigen::Vector<double, 6>& pose_rl);
private:
    CameraCalibrationSolver solver_l;
    CameraCalibrationSolver solver_r;
    std::vector<int> ids_;
private:
    template<typename CameraModelGeneratorT>
    class CalibrationStereoCF {
    public:
        template<typename T>
        bool operator()(
            const T camera_params_l[CameraModelGeneratorT::param_size_],
            const T camera_params_r[CameraModelGeneratorT::param_size_],
            const T pose_l[6], 
            const T pose_rl[6], 
            T residuals[4]) const 
        {
            const auto cm_l = CameraModelGeneratorT::Create(camera_params_l);
            const auto cm_r = CameraModelGeneratorT::Create(camera_params_r);

            const Eigen::Vector3<T> gloabal_point = object_point_.cast<T>();        
            const Eigen::Vector3<T> camera_point_l = TransformPoint(pose_l, gloabal_point.data());
            const Eigen::Vector2<T> projection_l = cm_l.Project(camera_point_l);
            
            const Eigen::Vector3<T> camera_point_r = TransformPoint(pose_rl, camera_point_l.data());
            const Eigen::Vector2<T> projection_r = cm_r.Project(camera_point_r);
            
            const Eigen::Vector2<T> error_l = T(weight_l_) * (projection_l - image_point_l_.cast<T>());
            residuals[0] = error_l[0];
            residuals[1] = error_l[1];

            const Eigen::Vector2<T> error_r = T(weight_r_) * (projection_r - image_point_r_.cast<T>());
            residuals[2] = error_r[0];
            residuals[3] = error_r[1];

            return true; 
        }

        static ceres::CostFunction* Create(
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point_l,
            const Eigen::Vector2d& image_point_r,
            double weight_l,
            double weight_r) {
                return new ceres::AutoDiffCostFunction<CalibrationStereoCF<CameraModelGeneratorT>, 4, CameraModelGeneratorT::param_size_, CameraModelGeneratorT::param_size_, 6, 6>(
                    new CalibrationStereoCF<CameraModelGeneratorT>(object_point, image_point_l, image_point_r, weight_l, weight_r));
            }

        CalibrationStereoCF(
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point_l,
            const Eigen::Vector2d& image_point_r,
            double weight_l,
            double weight_r) 
            : object_point_(object_point)
            , image_point_l_(image_point_l)
            , image_point_r_(image_point_r)
            , weight_l_(weight_l)
            , weight_r_(weight_r) 
            {}

    private:
        Eigen::Vector3d object_point_;
        Eigen::Vector2d image_point_l_;
        Eigen::Vector2d image_point_r_;
        double weight_l_;
        double weight_r_;
    };

    
};

template<typename CameraModelGeneratorT>
void StereoCameraCalibrationSolver::Calibrate(
    Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
    Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
    Eigen::Vector<double, 6>& pose_rl) 
{
    Eigen::Matrix<double, CameraModelGeneratorT::param_size_, CameraModelGeneratorT::param_size_> info_mat_l;
    Eigen::Matrix<double, CameraModelGeneratorT::param_size_, CameraModelGeneratorT::param_size_> info_mat_r;
    solver_l.Calibrate<CameraModelGeneratorT>(camera_params_l, info_mat_l); 
    solver_r.Calibrate<CameraModelGeneratorT>(camera_params_r, info_mat_r); 

    std::unordered_map<int, size_t> id_idx_l;
    std::unordered_map<int, size_t> id_idx_r;

    for(int i = 0; i < solver_l.frames_.size(); ++i) {
        id_idx_l[solver_l.frames_[i].id_] = i;
    }

    for(int i = 0; i < solver_r.frames_.size(); ++i) {
        id_idx_r[solver_r.frames_[i].id_] = i;
    }

    ceres::Problem problem;
    pose_rl.setZero();

    for(auto id : ids_) {
        auto itr_l = id_idx_l.find(id);
        auto itr_r = id_idx_r.find(id);

        if (id_idx_l.end() == itr_l || id_idx_r.end() == itr_r) {
            continue;
        }

        auto& frame_l = solver_l.frames_.at(itr_l->second);
        auto& frame_r = solver_r.frames_.at(itr_r->second);

        for(int i = 0; i < frame_l.object_points_.size(); ++i) {
            const Eigen::Vector3d& object_point = frame_l.object_points_[i];
            const Eigen::Vector2d& image_point_l = frame_l.image_points_[i];
            const Eigen::Vector2d& image_point_r = frame_r.image_points_[i];
            double weight_l = frame_l.weights_[i];
            double weight_r = frame_r.weights_[i];

            ceres::CostFunction* cost_function_l = CalibrationStereoCF<CameraModelGeneratorT>::Create(
                object_point,
                image_point_l, 
                image_point_r,
                weight_l,
                weight_r);

            problem.AddResidualBlock(cost_function_l, new ceres::CauchyLoss(2.0), 
                camera_params_l.data(), camera_params_r.data(), frame_l.pose_.data(), pose_rl.data());
        }

        problem.SetParameterBlockConstant(frame_l.pose_.data());
    }
    problem.SetParameterBlockConstant(camera_params_l.data());
    problem.SetParameterBlockConstant(camera_params_r.data());

    Optimize(problem, true, 1000, 1e-9);
}

/*
class StereoCameraCalibrationSolver {
public:
    using Point3D = Eigen::Vector3d;
    using Point2D = Eigen::Vector2d;

struct StereoFrame {
        std::vector<Eigen::Vector3d> pattern_points_; 
        std::vector<Eigen::Vector2d> image_points_l_;
        std::vector<Eigen::Vector2d> image_points_r_;
        std::vector<double> weights_;
        Eigen::Vector<double, 6> pose_;
    };
public:
     void AddFrame(
        const std::vector<Point3D>& object_points, 
        const std::vector<Point2D>& image_points_l,
        const std::vector<Point2D>& image_points_r,
        const std::vector<double>& weights) {
            frames_.emplace_back(object_points, image_points_l, image_points_r, weights);
        }   

    void AddFrame(
        const StereoFrame& frame) {
            frames_.push_back(frame);
        }   

    template<typename CameraModelGeneratorT>
    void Calibrate(
        const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
        const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
        Eigen::Vector<double, 6>& pose_rl);
private:
    void NormalizeSpatialDensity();

    std::vector<StereoFrame> frames_;
    static constexpr double cell_size_ = 20;
private:
    template<typename CameraModelGeneratorT>
    class CalibrationStereoCF {
    public:
        template<typename T>
        bool operator()(const T pose_lw[6], const T pose_rl[6], T residuals[4]) const {
            T camera_params_l[CameraModelGeneratorT::param_size_];
            T camera_params_r[CameraModelGeneratorT::param_size_];
            
            for(int i = 0; i < CameraModelGeneratorT::param_size_; ++i) {
              camera_params_l[i] = T(camera_params_l_[i]);  
              camera_params_r[i] = T(camera_params_r_[i]);  
            }
            
            const auto cm_l = CameraModelGeneratorT::Create(camera_params_l);
            const auto cm_r = CameraModelGeneratorT::Create(camera_params_r);

            const Eigen::Vector3<T> gloabal_point = object_point_.cast<T>();
            const Eigen::Vector3<T> camera_point_l = TransformPoint(pose_lw, gloabal_point.data());
            const Eigen::Vector2<T> projection_l = cm_l.Project(camera_point_l);

            const Eigen::Vector3<T> camera_point_r = TransformPoint(pose_rl, camera_point_l.data());
            const Eigen::Vector2<T> projection_r = cm_r.Project(camera_point_r);
            
            const Eigen::Vector2<T> error_l = (projection_l - image_point_l_.cast<T>());
            const Eigen::Vector2<T> error_r = (projection_r - image_point_r_.cast<T>());
            residuals[0] = error_l[0];
            residuals[1] = error_l[1];
            residuals[2] = error_r[2];
            residuals[3] = error_r[3];

            return true;
        }

        static ceres::CostFunction* Create(
            const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
            const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point_l,
            const Eigen::Vector2d& image_point_r) {
                return new ceres::AutoDiffCostFunction<CalibrationStereoCF<CameraModelGeneratorT>, 4, 6, 6>(
                    new CalibrationStereoCF<CameraModelGeneratorT>(camera_params_l, camera_params_r, object_point, image_point_l, image_point_r));
            }

        CalibrationStereoCF(
            const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
            const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
            const Eigen::Vector3d& object_point,
            const Eigen::Vector2d& image_point_l,
            const Eigen::Vector2d& image_point_r) 
            : camera_params_l_(camera_params_l)
            , camera_params_r_(camera_params_r)
            , object_point_(object_point)
            , image_point_l_(image_point_l)
            , image_point_r_(image_point_r){}

    private:
        Eigen::Vector<double, CameraModelGeneratorT::param_size_> camera_params_l_;
        Eigen::Vector<double, CameraModelGeneratorT::param_size_> camera_params_r_;
        Eigen::Vector3d object_point_;
        Eigen::Vector2d image_point_l_;
        Eigen::Vector2d image_point_r_;
    };

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
void StereoCameraCalibrationSolver::Calibrate(
    const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_l,
    const Eigen::Vector<double, CameraModelGeneratorT::param_size_>& camera_params_r,
    Eigen::Vector<double, 6>& pose_rl) 
{
    ceres::Problem problem;

    pose_rl.setZero();

    for(auto& frame : frames_) {
        frame.pose_ << 0, 0, 0, 0, 0, 1;
        for(size_t i = 0; i < frame.pattern_points_.size(); ++i) {
            const auto& pattern_point = frame.pattern_points_[i];
            const auto& image_point_l = frame.image_points_l_[i];
            const auto& image_point_r = frame.image_points_r_[i];
            ceres::CostFunction* cost_function_l = CalibrationStereoCF<CameraModelGeneratorT>::Create(camera_params_l, camera_params_r, pattern_point, image_point_l, image_point_r);
            problem.AddResidualBlock(cost_function_l, new ceres::CauchyLoss(2.0), frame.pose_.data(), pose_rl.data());
        }
    }

    Optimize(problem, true, 1000, 1e-9);
}
*/