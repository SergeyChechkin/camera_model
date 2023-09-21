#pragma once

#include <Eigen/Core>

template <typename T>
class FlatCheckerboardPattern {
public:
    Eigen::Vector2<T> pattern_board_size_;  // size of the pattern board
    Eigen::Vector2<T> checker_offset_;      // offset of the pattern from top-left corner
    Eigen::Vector2<T> checker_size_;        // size of the checker
    Eigen::Vector2i pattern_size_;          // size of the pattern, number of checkers
public:

    std::vector<Eigen::Vector3<T>> GetAllCorners() const {
        std::vector<Eigen::Vector3<T>> result;
        result.reserve((pattern_size_[0] + 1) * (pattern_size_[1] + 1));
        for(int v = 0; v <= pattern_size_[1]; ++v) {
            for(int u = 0; u <= pattern_size_[0]; ++u) {
                result.emplace_back(u * checker_size_[0] + checker_offset_[0], v * checker_size_[1] + checker_offset_[1], T(0));
            }
        }

        return result;
    }

    std::vector<Eigen::Vector3<T>> GetSaddleCorners() const {
        std::vector<Eigen::Vector3<T>> result;
        result.reserve((pattern_size_[0] - 1) * (pattern_size_[1] - 1));
        for(int v = 1; v < pattern_size_[1]; ++v) {
            for(int u = 1; u < pattern_size_[0]; ++u) {
                result.emplace_back(u * checker_size_[0] + checker_offset_[0], v * checker_size_[1] + checker_offset_[1], T(0));
            }
        }

        return result;
    }
};