/// BSD 3-Clause License
/// Copyright (c) 2023, Sergey Chechkin
/// Autor: Sergey Chechkin, schechkin@gmail.com 

#include "camera_model/ProjectionModel.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <random>


TEST(ProjectionTest, PerspectiveProjectionTest) { 
    PerspectivePM<double> projection;
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
    EquidistantPM<double> projection;
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}