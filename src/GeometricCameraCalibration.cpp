#include "camera_calibration/GeometricCameraCalibration.h"
#include <spatial_hash/SpatialHash2DVector.h>

void CameraCalibrationSolver::AddFrame(
    int id,
    const std::vector<Point3D>& object_points, 
    const std::vector<Point2D>& image_points,
    const std::vector<double>& weights)
{
    CHECK_EQ(object_points.size(), image_points.size());
    CHECK_EQ(object_points.size(), weights.size());
    
    Frame new_frame;
    new_frame.id_ = id;
    new_frame.object_points_ = object_points;
    new_frame.image_points_ = image_points;
    new_frame.weights_ = weights;
    new_frame.pose_ << 0, 0, 0, 0, 0, 1;

    frames_.push_back(new_frame);
}

void CameraCalibrationSolver::NormalizeSpatialDensity() 
{
    libs::spatial_hash::SpatialHashTable2DVector<double, std::pair<size_t, size_t>> hash_table(cell_size_); 

    static double radius_sqr = cell_size_ * cell_size_;

    size_t total_count = 0; 
    for(int i = 0; i < frames_.size(); ++i) {
        const auto& frame = frames_[i];  
        for(int j = 0; j < frame.image_points_.size(); ++j) {
            const auto& ip = frame.image_points_[j];            
            hash_table.Add(ip.data(), {i,j});
            ++total_count;
        }
    }

    double sum_of_weights = 0; 

    for(int i = 0; i < frames_.size(); ++i) {
        auto& frame = frames_[i];  
        for(int j = 0; j < frame.image_points_.size(); ++j) {
            const auto& ip = frame.image_points_[j]; 
            auto cell_idx = hash_table.GetCellIndex(ip.data());
            auto square_idxs = hash_table.SquareSearch(cell_idx, 1);   

            int count = 1;
            for(auto idx : square_idxs) {
                auto dif = frames_[idx.first].image_points_[idx.second] - ip;
                if (dif.dot(dif) < radius_sqr) {
                    ++count;
                }
            }

            const auto weight = 1.0 / count;
            frame.weights_[j] *= weight;
            sum_of_weights += weight;
        }
    }

    // normalize scales to 1 per point 
    double weights_scale = total_count / sum_of_weights; 
    
    for(int i = 0; i < frames_.size(); ++i) {
        auto& frame = frames_[i];  
        for(int j = 0; j < frame.image_points_.size(); ++j) {
            frame.weights_[j] *= weights_scale;
        }
    }
}