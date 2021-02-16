#include <iostream>
#include <sstream> 
#include <string>
#include <iterator>

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include <sophus/se3.hpp>

#include <filesystem>
#include <fstream>

#include <chrono>


#include <optimization/io.h>
#include <optimization/camera.h>
#include <optimization/residuals.h>

/* TODO: references?? const?? */
/* TODO: point and normal in camera view */
/* pose * point, pose.rotation_matrix() * normal */
Eigen::Vector3d get_plane(Eigen::Vector3d p, Eigen::Vector3d normal) {
    Eigen::Vector3d n{normal / normal[2]};
    double d = n.transpose().dot(p);
    n /= d;
    return n;
}

// TODO how to make it dependent on input
cv::Vec3d bilinear_interpolation(cv::Mat img, cv::Point2d pt) {
    int x = (int)pt.x;
    int y = (int)pt.y;
    float a = pt.x - (float)x;
    float c = pt.y - (float)y;
    cv::Vec3d v = (cv::Vec3d(img.at<cv::Vec3b>(x, y)) * (1.f - a) + 
                cv::Vec3d(img.at<cv::Vec3b>(x+1, y)) * a) * (1.f - c) + 
               (cv::Vec3d(img.at<cv::Vec3b>(x, y+1)) * (1.f - a) + 
                cv::Vec3d(img.at<cv::Vec3b>(x+1, y+1)) * a) * c;
   return v;
}

std::array<Eigen::Vector3d, 16> create_grid(Eigen::Vector3d center, 
                                         Eigen::Vector3d normal, double scale) {
    Eigen::Vector3d horizontal{-normal[2], 0, normal[0]};
    Eigen::Vector3d vertical{horizontal.cross(normal)};
    horizontal.normalize();
    vertical.normalize();
    
    std::array<Eigen::Vector3d, 16> grid;
    int index = 0;
    
    for(float y = -1.; y <= 1; y += 2./3.) {
        for(float x = -1.; x <= 1.; x += 2./3.) {
            grid[index] = (center + x * horizontal * scale + y * vertical * scale);
            index++;
        }
    }
    return grid;
}


int main() {
    std::filesystem::path camera_path{"../../../data/horse_workspace/camera.txt"};
    std::filesystem::path pose_path{"../../../data/horse_workspace/poses.txt"};
    std::filesystem::path visibility_path{"../../../data/horse_workspace/visibility_0.txt"};
    std::filesystem::path landmark_path{"../../../data/horse_workspace/landmarks_0.txt"};
    std::filesystem::path image_path{"../../../data/horse_workspace/images/"};
    
    std::vector<Sophus::SE3d> poses;
    read_poses(pose_path, poses);
    
    RadialCamera<double>::VecN camera_param;
    read_camera_parameters(camera_path, camera_param);
    RadialCamera<double> camera{camera_param};
    
    std::vector<std::vector<int>> visibility;
    read_visibility(visibility_path, visibility);
    
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> scales;
    read_points_normals_gridscales(landmark_path, points, normals, scales);
    
    std::vector<cv::Mat> images;
    for (int i = 0; i < poses.size(); i++) {
        images.push_back(cv::imread(image_path / 
                         (std::string((5 - std::to_string(i+1).length()), '0') +
                          std::to_string(i+1)  + ".jpg")));
    } 
    
    /* begin optimization */
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    /* construct grids */
    std::vector<std::array<Eigen::Vector3d, 16>> grids;
    for (int i = 0; i < points.size(); i++) {
        grids.push_back(create_grid(points[i], normals[i], scales[i]));
    }
    
    for (int point_idx = 0; point_idx < points.size(); point_idx ++) {
        std::vector<std::array<double, 16>> patches;
        std::array<double, 16> mean_patch;
        for (int i = 0; i < 16; i++) {
            mean_patch[i] = 0;
        }
        for (int v = 0; v < visibility[point_idx].size(); v++) {
            int view_idx = visibility[point_idx][v];
            std::array<Eigen::Vector2d, 16> p2d; 
            for (int i = 0; i < 16; i++) {
                p2d[i] = camera.project(poses[view_idx] * grids[point_idx][i]);
            }
            std::array<double, 16> patch;
            double mean = 0;
            double std = 0;
            for (int i = 0; i < 16; i++) {
                cv::Vec3d v3 = bilinear_interpolation(images[view_idx], {p2d[i][1], p2d[i][0]});
                patch[i] = (v3[0] + v3[1] + v3[2]) / 3.;
                mean += patch[i];
            }
            mean /= 16;
            for (int i = 0; i < 16; i++) {
                patch[i] -= mean;
                std += patch[i] * patch[i];
            }
            std = sqrt(std);
            for (int i = 0; i < 16; i++) {
                patch[i] /= std;
                mean_patch[i] += patch[i];
            }
            patches.push_back(patch);
        }
        
        std::array<double, 16> mu;
        for (int i = 0; i < 16; i++) {
            mean_patch[i] /= visibility[0].size();
            mu[i] = mean_patch[i];
        }
        
        ceres::Problem problem;
        for (int i = 0; i < patches.size(); i++) {
            ceres::CostFunction* cost = 
                new ceres::AutoDiffCostFunction<DifferenceCostFunctor, 16, 16>(new DifferenceCostFunctor(&(patches[i][0])));
            problem.AddResidualBlock(cost, new Rho(), &(mu[0]));
        }
        
        //ceres::Solver::Options::dogleg_type
        ceres::Solver::Options options;
        /* TODO: linear solver */
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        //options.dogleg_type = ceres::SUBSPACE_DOGLEG;
        
        options.logging_type = ceres::SILENT;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        
        if (point_idx % 500 == 0) {
            std::cout << point_idx << std::endl;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << std::endl;

}









