
#include <iostream>
#include <sstream> 
#include <string>
#include <iterator>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sophus/se3.hpp>

#include <filesystem>
#include <fstream>

#include <chrono>


#include <optimization/io.h>
#include <optimization/camera.h>
#include <optimization/residuals.h>


int main() {
    /* TODO: make static*/
    std::filesystem::path pose_path{"../../../data/horse_workspace/poses.txt"};
    std::filesystem::path image_path{"../../../data/horse_workspace/images/"};
    
    std::vector<Sophus::SE3d> poses;
    read_poses(pose_path, poses);
    
    /* TODO: move loading images to io file? */
    /* load images */
    std::vector<cv::Mat> images;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>*> interpolators;
    
    for (int i = 0; i < poses.size(); i++) {
        cv::Mat img{cv::imread((image_path / 
                                  (std::string((5 - std::to_string(i+1).length()), '0') +
                                  std::to_string(i+1)  + ".jpg")), cv::IMREAD_GRAYSCALE)};
        cv::Mat image;
        img.convertTo(image, CV_64F);
        images.push_back(image);
        
        grids.push_back(ceres::Grid2D<double>((double*)(images[i].data), 0, 1080, 0, 1920));
        
    }
    for (int i = 0; i < poses.size(); i++)  {
        interpolators.push_back(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grids[i]));
    }
    
    /* TODO: 3D point ? */
    /* */
}