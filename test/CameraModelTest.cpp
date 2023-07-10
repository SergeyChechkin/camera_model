/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "camera_model/ProjectionModel.h"
#include "camera_model/DistortionModel.h"
#include "camera_model/CameraModel.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <random>


TEST(ProjectionTest, PerspectiveProjectionTest) { 
    Perspective<double> projection(1);
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
    Equidistant<double> projection;
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

TEST(CameraModelTest, CameraModelTest) {
    const Eigen::Vector2d pp(320, 240);
    const double f = 500;
    std::shared_ptr<ProjectionModel<double>> projection = std::make_shared<Perspective<double>>(f);
    std::array<double, 6> params_6 = {0, 0, 0, 0, 0, 0};
    std::shared_ptr<DistortionModel<double>> distortion = std::make_shared<GenericCombined<double>>(params_6.data());
    UniversalCameraModel<double> cm(projection, distortion, pp);

    Eigen::Vector3d point_3d(0.1, 0.1, 1);
    auto image_point = cm.Project(point_3d); 
    ASSERT_DOUBLE_EQ(image_point[0], 370);
    ASSERT_DOUBLE_EQ(image_point[1], 290);

    auto up_point = cm.ReProjectToUnitPlane(image_point);

    ASSERT_DOUBLE_EQ(point_3d[0], up_point[0]);
    ASSERT_DOUBLE_EQ(point_3d[1], up_point[1]);
    ASSERT_DOUBLE_EQ(point_3d[2], up_point[2]);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}