/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "camera_model/ProjectionModel.h"
#include "camera_model/DistortionModel.h"
#include "camera_model/GeometricCameraModel.h"
#include "camera_model/PinholeCameraModel.h"

#include "camera_calibration/CameraModelGenerator.h"
#include "camera_calibration/GeometricCameraModelCalibration.h"
#include "camera_calibration/CameraCalibration.h"

#include <utils/IOStreamUtils.h>
#include "utils/CeresUtils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <random>


TEST(ProjectionTest, PerspectiveProjectionTest) { 
    double f = 1.0;
    Perspective<double> projection(&f);
    Eigen::Vector3d point_3d(1,1,1);
    auto point_2d = projection.Project(point_3d);

    ASSERT_DOUBLE_EQ(point_2d[0], 1.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 1.0);

    auto point_3d_ = projection.ReProjectToUnitPlane(point_2d);

    ASSERT_DOUBLE_EQ(point_3d_[0], 1.0);
    ASSERT_DOUBLE_EQ(point_3d_[1], 1.0);
    ASSERT_DOUBLE_EQ(point_3d_[2], 1.0);

    point_3d.normalize();
    point_3d_ = projection.ReProjectToUnitSphere(point_2d);

    ASSERT_DOUBLE_EQ(point_3d_[0], point_3d[0]);
    ASSERT_DOUBLE_EQ(point_3d_[1], point_3d[1]);
    ASSERT_DOUBLE_EQ(point_3d_[2], point_3d[2]);

    point_2d = projection.Project({0,0,0});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_2d = projection.Project({0,0,1});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_2d = Eigen::Vector2d(0, 0);
    point_3d_ = projection.ReProjectToUnitPlane(point_2d);

    ASSERT_DOUBLE_EQ(point_3d_[0], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[1], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[2], 1.0);
}

TEST(ProjectionTest, EquidistantProjectionTest) { 
    double f = 1.0;
    Equidistant<double> projection(&f);
    Eigen::Vector3d point_3d(1,1,1); 
    point_3d.normalize();
    auto point_2d = projection.Project(point_3d);
    auto point_3d_ = projection.ReProjectToUnitSphere(point_2d);

    ASSERT_DOUBLE_EQ(point_3d_[0], point_3d[0]);
    ASSERT_DOUBLE_EQ(point_3d_[1], point_3d[1]);
    ASSERT_DOUBLE_EQ(point_3d_[2], point_3d[2]);

    point_2d = projection.Project({0,0,0});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_2d = projection.Project({0,0,1});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_2d = Eigen::Vector2d(0, 0);
    point_3d_ = projection.ReProjectToUnitSphere(point_2d);

    ASSERT_DOUBLE_EQ(point_3d_[0], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[1], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[2], 1.0);

    point_2d = M_PI / 2 * Eigen::Vector2d(1, 1).normalized();

    auto sperical = projection.ToSpherical(point_2d);

    ASSERT_DOUBLE_EQ(sperical[0], M_PI / 2);
    ASSERT_DOUBLE_EQ(sperical[1], M_PI / 4);
}

TEST(ProjectionTest, FoVProjectionTest) { 
    double f = 450.0;
    double w = 0.95;
    FieldOfView<double> projection(&f);
    Eigen::Vector3d point_3d(1,1,1); 
    auto point_2d = projection.Project(point_3d);
    auto point_3d_ = projection.ReProjectToUnitPlane(point_2d);

    ASSERT_NEAR(point_3d_[0], point_3d[0], 1.0e-6);
    ASSERT_NEAR(point_3d_[1], point_3d[1], 1.0e-6);
    ASSERT_NEAR(point_3d_[2], point_3d[2], 1.0e-6);

    point_2d = projection.Project({0,0,0});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_2d = projection.Project({0,0,1});

    ASSERT_DOUBLE_EQ(point_2d[0], 0.0);
    ASSERT_DOUBLE_EQ(point_2d[1], 0.0);

    point_3d_ = projection.ReProjectToUnitSphere({0,0});

    ASSERT_DOUBLE_EQ(point_3d_[0], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[1], 0.0);
    ASSERT_DOUBLE_EQ(point_3d_[2], 1.0);
}

TEST(DistortionTest, DistortionsTest) {
    std::array<double, 2> params = {0, 0}; 
    Eigen::Vector2d point(1,1); 
    Eigen::Vector2d result;

    RadialPolynomial<double, 2> dist_model_1(params.data());
    result = dist_model_1.Distort(point);
    ASSERT_DOUBLE_EQ(result[0], 0.0);
    ASSERT_DOUBLE_EQ(result[1], 0.0);

    Decentering<double> dist_model_2(params.data());
    result = dist_model_2.Distort(point);
    ASSERT_DOUBLE_EQ(result[0], 0.0);
    ASSERT_DOUBLE_EQ(result[1], 0.0);

    std::array<double, 4> params_4 = {0, 0, 0, 0};
    ThinPrism<double> dist_model_3(params_4.data());
    result = dist_model_3.Distort(point);
    ASSERT_DOUBLE_EQ(result[0], 0.0);
    ASSERT_DOUBLE_EQ(result[1], 0.0);

    std::array<double, 6> params_6 = {0, 0, 0, 0, 0, 0};
    GenericCombined<double> dist_model_4(params_6.data());
    result = dist_model_4.Distort(point);
    ASSERT_DOUBLE_EQ(result[0], 0.0);
    ASSERT_DOUBLE_EQ(result[1], 0.0);
}

TEST(CameraModelTest, GeometricCameraModelTest) {
    double f = 500;
    std::shared_ptr<ProjectionModel<double>> projection = std::make_shared<Perspective<double>>(&f);
    std::array<double, 6> params_6 = {0, 0, 0, 0, 0, 0};
    std::shared_ptr<DistortionModel<double>> distortion = std::make_shared<GenericCombined<double>>(params_6.data());
    GeometricCameraModel<double> cm(projection, distortion, {320, 240}, {640, 480});

    Eigen::Vector3d point_3d(0.1, 0.1, 1);
    auto image_point = cm.Project(point_3d); 
    ASSERT_DOUBLE_EQ(image_point[0], 370);
    ASSERT_DOUBLE_EQ(image_point[1], 290);

    auto up_point = cm.ReProjectToUnitPlane(image_point);

    ASSERT_DOUBLE_EQ(point_3d[0], up_point[0]);
    ASSERT_DOUBLE_EQ(point_3d[1], up_point[1]);
    ASSERT_DOUBLE_EQ(point_3d[2], up_point[2]);
}

TEST(CameraModelTest, GeometricCameraModelT_Test) {
    double f = 500;
    std::array<double, 6> params_6 = {0, 0, 0, 0, 0, 0};
    Eigen::Vector2d pp(320, 240);
    GeometricCameraModelT<double, Perspective<double>, GenericCombined<double>> cm(&f, params_6.data(), pp.data());

    Eigen::Vector3d point_3d(0.1, 0.1, 1);
    auto image_point = cm.Project(point_3d); 
    ASSERT_DOUBLE_EQ(image_point[0], 370);
    ASSERT_DOUBLE_EQ(image_point[1], 290);

    auto up_point = cm.ReProjectToUnitPlane(image_point);

    ASSERT_DOUBLE_EQ(point_3d[0], up_point[0]);
    ASSERT_DOUBLE_EQ(point_3d[1], up_point[1]);
    ASSERT_DOUBLE_EQ(point_3d[2], up_point[2]);
}

TEST(CameraModelTest, GeometricCameraCalibrationTest) {
    CameraCalibrationT calib;

    double f = 500;
    std::array<double, 6> params_3 = {0, 0, 0};
    Eigen::Vector2d pp(320, 240);    
    auto cm = PerspectiveOnlyGenerator::Create(&f, params_3.data(), pp.data());

    Eigen::Isometry3d pose;
    pose.translation() = Eigen::Vector3d(0, 0, 10);
    pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();

    {
        std::vector<Eigen::Vector3d> object_points; 
        std::vector<Eigen::Vector2d> image_points;
        std::vector<double> weights;

        for(int i = -20; i < 20 ; ++i) {
            for(int j = -20; j < 20 ; ++j) {
                object_points.emplace_back(i * 0.2, j * 0.2, 0);
                const auto cam_point = pose * object_points.back();
                const auto img_point = cm.Project(cam_point);
                image_points.push_back(img_point);
                weights.push_back(1.0);
            }
        }

        calib.AddFrame(object_points, image_points, weights);
    }

    pose.translation() = Eigen::Vector3d(0, 0, 50);
    pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();

    {
        std::vector<Eigen::Vector3d> object_points; 
        std::vector<Eigen::Vector2d> image_points;
        std::vector<double> weights;

        for(int i = -20; i < 20 ; ++i) {
            for(int j = -20; j < 20 ; ++j) {
                object_points.emplace_back(i * 0.2, j * 0.2, 0);
                const auto cam_point = pose * object_points.back();
                const auto img_point = cm.Project(cam_point);
                image_points.push_back(img_point);
                weights.push_back(1.0);
            }
        }

        calib.AddFrame(object_points, image_points, weights);
    }

    Eigen::Vector<double, PerspectiveOnlyGenerator::param_size_> params; 
    Eigen::Vector<double, PerspectiveOnlyGenerator::param_size_> info;

    params.setZero();
    params[0] = 450;
    params[1] = 330;
    params[2] = 230;

    calib.Calibrate<PerspectiveOnlyGenerator>(params, info);

    ASSERT_NEAR(params[0], f, 1.0e-5);
    ASSERT_NEAR(params[1], pp[0], 1.0e-5);
    ASSERT_NEAR(params[2], pp[1], 1.0e-5);
}

TEST(CameraModelTest, GeometricCameraCalibrationTest2) {
    cv::Size image_size(640, 480);
    cv::Size pattern_size(7, 6);
    PinholeCameraCalibration calibration(image_size, pattern_size);

    double f = 500;
    std::array<double, 6> params_3 = {0, 0, 0};
    Eigen::Vector2d pp(320, 240);    
    auto cm = PerspectiveOnlyGenerator::Create(&f, params_3.data(), pp.data());

    Eigen::Isometry3d pose;
    pose.setIdentity();
    pose.translation() = Eigen::Vector3d(0, 0, 10);
    pose.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();

    const double pose_[6] = {0, 0, 0, 0, 0, 10};

    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;

    for(int i = -20; i < 20 ; ++i) {
        for(int j = -20; j < 20 ; ++j) {
            Eigen::Vector3d object_point(i * 0.2, j * 0.2, 0);
            const auto cam_point = pose * object_point;
            const auto img_point = cm.Project(cam_point);

            image_points.emplace_back(img_point[0], img_point[1]);
            object_points.emplace_back(object_point[0], object_point[1], object_point[2]);
        }
    }

    calibration.AddFrame(image_points, object_points);
    std::array<double, 10> camera_params = {510, 510, 0.5 * image_size.width - 5, 0.5 * image_size.height - 5, 0, 0, 0, 0, 0, 0};
    calibration.Calibrate(camera_params);
    std::cout << camera_params << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}