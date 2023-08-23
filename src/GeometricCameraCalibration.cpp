#include "camera_calibration/GeometricCameraCalibration.h"

void CameraCalibrationT::AddFrame(
    const std::vector<Point3D>& object_points, 
    const std::vector<Point2D>& image_points,
    const std::vector<double>& weights)
{
    CHECK_EQ(object_points.size(), image_points.size());
    CHECK_EQ(object_points.size(), weights.size());
    
    Frame new_frame;
    new_frame.object_points_ = object_points;
    new_frame.image_points_ = image_points;
    new_frame.weights_ = weights;
    new_frame.pose_ << 0, 0, 0, 0, 0, 1;

    frames_.push_back(new_frame);
}