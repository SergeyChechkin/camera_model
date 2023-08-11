/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "camera_calibration/CameraCalibration.h"
#include "camera_model/PinholeCameraModel.h"
#include "utils/ImageUtils.h"
#include "utils/CeresUtils.h"

#include "spatial_hash/SpatialHash2DVector.h"
using namespace libs::spatial_hash;

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

PinholeCameraCalibration::PinholeCameraCalibration(cv::Size image_size, cv::Size pattern_size, cv::Size patch_size) 
: image_size_(image_size)
, pattern_size_(pattern_size)
, patch_size_(patch_size)
, flags_(cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK)
{ }

bool PinholeCameraCalibration::AddFrame(const cv::Mat& frame, cv::Mat* result) {
    cv::Mat gray_frame = ConvertToGray(frame);
        
    std::vector<cv::Point2f> detected_corners;
    bool sucsess = cv::findChessboardCorners(gray_frame, pattern_size_, detected_corners, flags_);
    
    if (sucsess) {
        cv::TermCriteria term_crit(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray_frame, detected_corners, patch_size_, cv::Size(-1, -1), term_crit);

        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point3f> object_points;
        std::vector<float> weights;

        auto itr = detected_corners.begin();
        for(int y = 0; y < pattern_size_.height; ++y) {
            for(int x = 0; x < pattern_size_.width; ++x) {
                image_points.emplace_back(itr->x + 0.5f, itr->y + 0.5f); 
                object_points.emplace_back(x, y, 0); 
                ++itr;
            } 
        } 

        image_points_.push_back(image_points);
        object_points_.push_back(object_points); 
        points_weights_.push_back(std::vector<float>(image_points.size(), 1));
        poses_.push_back({0, 0, 0, 0, 0, 1.0});
    }

    if (nullptr != result) {
        *result = ConvertToColor(gray_frame);
        cv::drawChessboardCorners(*result, pattern_size_, detected_corners, sucsess); 
    }

    return sucsess;
}

void PinholeCameraCalibration::CalculateWeight() {
    float cell_size = 20;
    SpatialHashTable2DVector<float, std::pair<size_t, size_t>> hash_table(cell_size); 
    
    size_t total_count = 0; 
    for(int i = 0; i < image_points_.size(); ++i) {
        for(int j = 0; j < image_points_[i].size(); ++j) {
            const auto& ip = image_points_[i][j];            
            hash_table.Add(&(ip.x), {i,j});
            ++total_count;
        }
    }

    float pixel_count = image_size_.height * image_size_.width;
    float count_per_pixel = total_count / pixel_count;

    for(int i = 0; i < image_points_.size(); ++i) {
        for(int j = 0; j < image_points_[i].size(); ++j) {
            const auto& ip = image_points_[i][j]; 
            auto cell_idx = hash_table.GetCellIndex(&(ip.x));
            auto  square_idxs = hash_table.SquareSearch(cell_idx, 1);   

            float radius_sqr = cell_size * cell_size;
            int count = 1;
            for(auto idx : square_idxs) {
                auto dif = image_points_[idx.first][idx.second]-ip;
                if (dif.dot(dif) < radius_sqr) {
                    ++count;
                }
            }

            points_weights_[i][j] = count_per_pixel * M_PI * radius_sqr / count;
        }
    }
}

template<typename T>
Eigen::Vector3<T> GlobalToCamera(const T src[3], const T pose[6]) {
    Eigen::Vector3<T> result;
    ceres::AngleAxisRotatePoint(pose, src, result.data());
    return result + Eigen::Vector3<T>(pose + 3);
}

template<typename T, typename CameraT, size_t Nm>
Eigen::Vector2<T> GlobalToImage(const T src[3], const T pose[6], const T camera_params[Nm]) {
    Eigen::Vector3<T> camera_point = GlobalToCamera(src, pose);
    CameraT cm(camera_params);
    return cm.Project(camera_point);
}

class PnpCalibCostFunction {
public:
    template<typename T>
    bool operator()(const T pose[6], const T camera_params[10], T residuals[2]) const {
        Eigen::Vector3<T> point_3D_pattern = oblect_point_.cast<T>();
        Eigen::Vector2<T> projection = GlobalToImage<T, PinholeCameraModel<T>, 10>(point_3D_pattern.data(), pose, camera_params); 
        Eigen::Vector2<T> error = T(weight_) * (projection - image_point_.cast<T>());
        residuals[0] = error[0];
        residuals[1] = error[1];

        return true;
    }

    static ceres::CostFunction* Create(
        const Eigen::Vector3d& oblect_point,
        const Eigen::Vector2d& image_point,
        double weight = 1.0) {
            return new ceres::AutoDiffCostFunction<PnpCalibCostFunction, 2, 6, 10>(
                new PnpCalibCostFunction(oblect_point, image_point, weight));
        }

    PnpCalibCostFunction(
        const Eigen::Vector3d& oblect_point,
        const Eigen::Vector2d& image_point,
        double weight = 1.0) 
        : oblect_point_(oblect_point)
        , image_point_(image_point)
        , weight_(weight){}

private:
    Eigen::Vector3d oblect_point_;
    Eigen::Vector2d image_point_;
    double weight_;
};

void PinholeCameraCalibration::Calibrate(std::array<double, 10>& camera_params) {
    CalculateWeight();
    /// Estimating camera poses for each frame with initial intrinisc guess.
    for(int i = 0; i < object_points_.size(); ++i) {
        ceres::Problem problem;
        for(int j = 0; j < object_points_[i].size(); ++j) {
            const auto& op = object_points_[i][j];
            const auto& ip = image_points_[i][j];            
            Eigen::Vector3d oblect_point(op.x, op.y, op.z);
            Eigen::Vector2d image_point(ip.x, ip.y);
            ceres::CostFunction* cost_function = PnpCalibCostFunction::Create(oblect_point, image_point);

            problem.AddResidualBlock(cost_function, nullptr, poses_[i].data(), camera_params.data());
        }
        problem.SetParameterBlockConstant(camera_params.data());
        Optimize(problem, false, 10, 1e-3);
    }

    /// solve camera calibration
    ceres::Problem problem;
    for(int i = 0; i < object_points_.size(); ++i) {
        for(int j = 0; j < object_points_[i].size(); ++j) {
            const auto& op = object_points_[i][j];
            const auto& ip = image_points_[i][j];
            Eigen::Vector3d oblect_point(op.x, op.y, op.z);
            Eigen::Vector2d image_point(ip.x, ip.y);
            ceres::CostFunction* cost_function = PnpCalibCostFunction::Create(oblect_point, image_point, points_weights_[i][j]);

            problem.AddResidualBlock(cost_function, nullptr, poses_[i].data(), camera_params.data());
        }
    }

    Optimize(problem, true, 1000, 1e-12);
}

PinholeCameraRemap::PinholeCameraRemap(
        const std::array<double, 10>& src_camera_params, 
        const std::array<double, 10>& dst_camera_params) 
        : src_cm_(src_camera_params.data()) 
        , dst_cm_(dst_camera_params.data()) {
        }

Eigen::Vector2d PinholeCameraRemap::RemapPoint(const Eigen::Vector2d& src) {
        return dst_cm_.Project(src_cm_.ReProject(src));
}

void PinholeCameraRemap::GenerateDistortionRemap(
        cv::Size dst_image_size,
        const Eigen::Matrix3d& rot,
        cv::Mat& dx, 
        cv::Mat& dy) {    
    dx = cv::Mat(dst_image_size.height, dst_image_size.width, CV_32F);
    dy = cv::Mat(dst_image_size.height, dst_image_size.width, CV_32F);
    
    cv::parallel_for_(cv::Range(0, dst_image_size.height), [&](const cv::Range & range){
        for(int v = range.start; v < range.end; ++v) {
            for(int u = 0; u < dst_image_size.width; ++u) {
                Eigen::Vector2d src(u + 0.5, v + 0.5);
                Eigen::Vector3d src_3d = rot * dst_cm_.ReProject(src);
                Eigen::Vector2d dst = src_cm_.Project(src_3d);
                dx.at<float>(v, u) = dst.x();
                dy.at<float>(v, u) = dst.y();
            }
        }
    });     
}