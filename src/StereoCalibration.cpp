#include <camera_calibration/StereoCalibration.h>
#include <camera_calibration/CalibrationPatternDetector.h>
#include <utils/image/ColorConvertion.h>
#include <utils/PoseUtils.h>

StereoCalibration::StereoCalibration(const FlatCheckerboardPattern<double>& pattern) 
: pattern_(pattern) 
{
}

cv::Mat StereoCalibration::DetectPattern(
    const cv::Mat& gray_image_l, 
    const cv::Mat& gray_image_r, 
    std::vector<Eigen::Vector3d>& pattern_points, 
    std::vector<Eigen::Vector2d>& image_points_l,
    std::vector<Eigen::Vector2d>& image_points_r) 
{
    auto l_status = FlatCheckerboardDetector_OpenCV<double>::DetectPattern(
        gray_image_l,
        pattern_,
        pattern_points, 
        image_points_l);

    auto r_status = FlatCheckerboardDetector_OpenCV<double>::DetectPattern(
        gray_image_r,
        pattern_,
        pattern_points, 
        image_points_r);  

    cv::Mat color_frame_l = gray_image_l.clone();
    cv::Mat color_frame_r = gray_image_r.clone();
    if (l_status && r_status) {
        color_frame_l = ConvertToColor(gray_image_l);
        color_frame_r = ConvertToColor(gray_image_r);

        FlatCheckerboardDetector_OpenCV<double>::DrawDetectedPattern(color_frame_l, pattern_, image_points_l);
        FlatCheckerboardDetector_OpenCV<double>::DrawDetectedPattern(color_frame_r, pattern_, image_points_r);
    }    

    return ImageCat(color_frame_l, color_frame_r);
}


cv::Mat StereoCalibration::AddFrame(int id, const cv::Mat& gray_image_l, const cv::Mat& gray_image_r, bool visualize) 
{
    std::vector<Eigen::Vector3d> pattern_points; 
    std::vector<Eigen::Vector2d> image_points_l;
    std::vector<Eigen::Vector2d> image_points_r;

    auto l_status = FlatCheckerboardDetector_OpenCV<double>::DetectPattern(
        gray_image_l,
        pattern_,
        pattern_points, 
        image_points_l);

    auto r_status = FlatCheckerboardDetector_OpenCV<double>::DetectPattern(
        gray_image_r,
        pattern_,
        pattern_points, 
        image_points_r);

    solver_stereo_.AddFrame(id, pattern_points, image_points_l, image_points_r);

    if (!visualize)
        return cv::Mat();

    // visualisation
    cv::Mat color_frame_l = gray_image_l.clone();
    cv::Mat color_frame_r = gray_image_r.clone();
    if (l_status && r_status) {
        color_frame_l = ConvertToColor(gray_image_l);
        color_frame_r = ConvertToColor(gray_image_r);

        FlatCheckerboardDetector_OpenCV<double>::DrawDetectedPattern(color_frame_l, pattern_, image_points_l);
        FlatCheckerboardDetector_OpenCV<double>::DrawDetectedPattern(color_frame_r, pattern_, image_points_r);
    }    

    return ImageCat(color_frame_l, color_frame_r);
}
