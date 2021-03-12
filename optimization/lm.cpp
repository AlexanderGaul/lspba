#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>

#include <sophus/se3.hpp>


#include <optimization/loader.h>
#include <optimization/io.h>
#include <optimization/camera.h>
#include <optimization/residuals.h>
#include <optimization/local_parameterization_se3.h>
#include <optimization/utils.h>


int main() {
    /* TODO: make static*/
    std::string camera_path{"../../../data/horse_workspace/preprocessing/camera.txt"};
    std::string pose_path{"../../../data/horse_workspace/preprocessing/poses.txt"};
    std::string image_path{"../../../data/horse_workspace/images/"};
    std::string landmark_path{"../../../data/horse_workspace/preprocessing/landmarks.txt"};
    std::string source_path{"../../../data/horse_workspace/preprocessing/source_views.txt"};
    std::string visibility_path{"../../../data/horse_workspace/preprocessing/visibility.txt"};
    std::string points3d_path{"../../../data/horse_workspace/preprocessing/points_normals_gridscales_select.txt"};
    
    std::string output_path{"../../../data/horse_workspace/optimization/points_normals.txt"};
    
    RadialCamera<double>::VecN camera_param;
    std::vector<Sophus::SE3d> poses;
    std::vector<Eigen::Vector2d> pixels;
    std::vector<Eigen::Vector3d> planes;
    std::vector<int> source_views;
    std::vector<std::vector<int>> visibilities;
    std::vector<Eigen::Vector3d> points_3D;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double> scales;
    
    read_camera_parameters(camera_path, camera_param);
    read_poses(pose_path, poses);
    read_landmarks(landmark_path, pixels, planes);
    read_source_frame(source_path, source_views);
    read_visibility(visibility_path, visibilities);
    read_points_normals_gridscales(points3d_path, points_3D, normals, scales);
    
    
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow();
    std::shared_ptr<open3d::geometry::PointCloud> cloud{
        new open3d::geometry::PointCloud(points_3D)};
    vis.AddGeometry(cloud);
    vis.Run();
    
    
    
    
    int observations = 0;
    for (int i = 0; i < visibilities.size(); i++) {
        observations += visibilities[i].size();
    }
    std::cout << "Landmarks: " << pixels.size() << std::endl;
    std::cout << "Observations: " << observations << std::endl;
    
    
    std::vector<cv::Mat> images;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    
    // Read images
    for (int i = 0; i < poses.size(); i++) {
        std::cout << i << std::endl;
        cv::Mat img{cv::imread((image_path / 
                                  (std::string((5 - std::to_string(i+1).length()), '0') +
                                  std::to_string(i+1)  + ".jpg")), cv::IMREAD_GRAYSCALE)};
        std::cout << "read" << std::endl;
        cv::Mat image;
        img.convertTo(image, CV_64F);
        std::cout << "converted" << std::endl;
        images.push_back(image);
        cv::imshow("hi", img);
        cv::waitKey(0);
        //std::cout << image.data << std::endl;
        //std::cout << images[i].data << std::endl;
        grids.push_back(ceres::Grid2D<double>((double*)(images[i].data), 0, 1080, 0, 1920));
    }
    for (int i = 0; i < poses.size(); i++)  {
        interpolators.push_back(ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grids[i]));
    }

    RadialCamera<double> camera{camera_param};

    
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches;
    
    for (int l = 0; l < pixels.size(); l++) {
        Eigen::Matrix<double, 2, 16> grid;
        Eigen::Matrix<double, 16, 1> patch;
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                grid.col(i)[0] = pixels[l][0] + x;
                grid.col(i)[1] = pixels[l][1] + y;
                interpolators[source_views[l]].Evaluate(grid.col(i)[1], 
                                                        grid.col(i)[0],
                                                        &patch[i]);
                
                i++;
            }
        }
        source_pixels.push_back(grid);
        Eigen::Matrix<double, 16, 1> patch_normalized{
            (patch - Eigen::Matrix<double, 16, 1>::Ones() * (patch.sum() / 16.)) /
            (patch - Eigen::Matrix<double, 16, 1>::Ones() * (patch.sum() / 16.)).norm()};
        source_patches.push_back(patch_normalized);
    }
    
    ceres::Problem problem;
    
    std::vector<bool> pose_added;
    for (int i = 0; i < poses.size(); i++) {
        pose_added.push_back(false);
    }
    
    for (int l = 0; l < pixels.size(); l++) {
        for (int v = 0; v < visibilities[l].size(); v++) {
            if (visibilities[l][v] == source_views[l]) continue;
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<PatchResidual, 1, 
                                                Sophus::SE3d::num_parameters, 
                                                Sophus::SE3d::num_parameters,
                                                6,  
                                                3>(
                    new PatchResidual(source_patches[l].data(), 
                                      source_pixels[l].data(),
                                      &interpolators[source_views[l]]));
            problem.AddResidualBlock(cost, new Rho(), poses[source_views[l]].data(),
                                                      poses[visibilities[l][v]].data(),
                                                      camera_param.data(),
                                                      planes[l].data());
            pose_added[visibilities[l][v]] = true;
            pose_added[source_views[l]] = true;
        }
    }
    std::cout << "Residuals added" << std::endl;
    int num_cameras = 0;
    for (int i = 0; i < poses.size(); i++) {
        if (pose_added[i]) {
        problem.AddParameterBlock(poses[i].data(), Sophus::SE3d::num_parameters,
                                  new Sophus::LocalParameterizationSE3);
            num_cameras++;
        }
        
    }
    
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.use_explicit_schur_complement = false;
    options.num_threads = 4;
    options.max_num_iterations = 5;

    ceres::Solver::Summary summary;
    
    Solve(options, &problem, &summary);
    
    std::cout << summary.FullReport() << std::endl;
    
    std::cout << camera_param << std::endl;
    RadialCamera<double> camera_optimized(camera_param);
    
    std::vector<Eigen::Vector3d> points_optimized;
    for (int i = 0; i < pixels.size(); i++) {
        Eigen::Vector3d X = unproject(pixels[i], planes[i], camera_optimized, poses[source_views[i]]);
        points_optimized.push_back(X);
    }
    
    
    /* TODO: store results somewhere */
}




