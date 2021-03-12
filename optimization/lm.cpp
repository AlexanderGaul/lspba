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


    
    
    
    int observations = 0;
    for (int i = 0; i < visibilities.size(); i++) {
        observations += visibilities[i].size();
    }
    std::cout << "Landmarks: " << pixels.size() << std::endl;
    std::cout << "Observations: " << observations << std::endl;
    
    std::cout << visibilities.size() << std::endl;
    
    std::vector<cv::Mat> images;
    std::vector<open3d::geometry::ImagePyramid> o3d_pyramids;
    //std::vector<std::array<cv::Mat, 2>> pyramids;
    std::vector<cv::Mat> mats_0;
    std::vector<cv::Mat> mats_1;
    std::vector<std::array<ceres::Grid2D<double, 1>, 2>> grid_pyramids;  /* TODO: rename grid pyramids*/
    std::vector<std::array<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>, 2>> interpolator_pyramids;
    /* TODO: reserve space in vectors */
    // Read images
    for (int i = 0; i < poses.size(); i++) {
        std::string im_file = image_path + (std::string((5 - std::to_string(i+1).length()), '0') + std::to_string(i+1) + ".jpg");
        
        open3d::geometry::Image o3d_im_color;
        open3d::io::ReadImage(im_file, o3d_im_color);
        std::shared_ptr<open3d::geometry::Image> o3d_im_pointer{
            o3d_im_color.CreateFloatImage(open3d::geometry::Image::ColorToIntensityConversionType::Equal)};
        open3d::geometry::ImagePyramid pyramid = o3d_im_pointer->CreatePyramid(2);
        o3d_pyramids.push_back(pyramid);
        
        cv::Mat im_float_1(1080, 1920, CV_32FC1, o3d_pyramids[i][0]->PointerAs<float>());
        cv::Mat im_float_2(o3d_pyramids[i][1]->height_, o3d_pyramids[i][1]->width_,
                           CV_32FC1, o3d_pyramids[i][1]->PointerAs<float>());
        cv::Mat mat1;
        cv::Mat mat2;
        im_float_1.convertTo(mat1, CV_64F);
        im_float_2.convertTo(mat2, CV_64F);
        mats_0.push_back(mat1);
        mats_1.push_back(mat2);
    }
    for (int i = 0; i < poses.size(); i++) {
        grid_pyramids.push_back({ceres::Grid2D<double, 1>((double*)(mats_0[i].data), 0, mats_0[i].rows, 0, mats_0[i].cols),
                                 ceres::Grid2D<double, 1>((double*)(mats_1[i].data), 0, mats_1[i].rows, 0, mats_1[i].cols)});
    }
    for (int i = 0; i < poses.size(); i++) {
        interpolator_pyramids.push_back({ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids[i][0]),
                                         ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids[i][1])});
    }


    RadialCamera<double> camera{camera_param};

    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches;
    /* Setting fo source parameters  for scale 2 */
    for (int l = 0; l < pixels.size(); l++) {
        Eigen::Matrix<double, 2, 16> grid;
        Eigen::Matrix<double, 16, 1> patch;
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                grid.col(i)[0] = pixels[l][0] + 2 * x;
                grid.col(i)[1] = pixels[l][1] + 2 + y;
                if (grid.col(i)[0] > 1920 || grid.col(i)[0] < 0) std::cout << grid.col(i) << std::endl;
                if (grid.col(i)[1] > 1080 || grid.col(i)[1] < 0) std::cout << grid.col(i) << std::endl;
                interpolator_pyramids[source_views[l]][1].Evaluate(grid.col(i)[1] * 0.5, 
                                                                   grid.col(i)[0] * 0.5,
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
    
    std::vector<PatchResidual*> residuals;
    
    std::vector<Eigen::Vector3d> points_selected;
    for (int l = 0; l < pixels.size(); l++) {
        if (l % 5 != 0) continue;
        Eigen::Vector3d X = unproject(pixels[l], planes[l], camera, poses[source_views[l]]);
        points_selected.push_back(X);
        
        for (int v = 0; v < visibilities[l].size(); v++) {
            if (visibilities[l][v] == source_views[l]) continue;
            residuals.push_back(new PatchResidual(source_patches[l].data(), 
                                                  source_pixels[l].data(),
                                                  &interpolator_pyramids[source_views[l]][1]));
            residuals.back()->scale = 0.5;
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<PatchResidual, 16,
                                                Sophus::SE3d::num_parameters, 
                                                Sophus::SE3d::num_parameters, 
                                                6, 
                                                3>(residuals.back());
                    
            problem.AddResidualBlock(cost, new Rho(), poses[source_views[l]].data(),
                                                      poses[visibilities[l][v]].data(),
                                                      camera_param.data(),
                                                      planes[l].data());
            pose_added[visibilities[l][v]] = true;
            pose_added[source_views[l]] = true;
        }
    }
    /* TODO: add soft constraint on distance between two cameras */
    ceres::CostFunction* cost = 
        new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 
                                        Sophus::SE3d::num_parameters, 
                                        Sophus::SE3d::num_parameters>(new DistanceCostFunctor((poses[58].translation() - 
                                                                                               poses[85].translation()).norm()));
    problem.AddResidualBlock(cost, nullptr, poses[58].data(), poses[85].data());
    
    
    std::cout << "Residuals added" << std::endl;
    int num_cameras = 0;
    for (int i = 0; i < poses.size(); i++) {
        if (pose_added[i]) {
        problem.AddParameterBlock(poses[i].data(), Sophus::SE3d::num_parameters,
                                  new Sophus::LocalParameterizationSE3);
            num_cameras++;
        }
    }
    problem.SetParameterBlockConstant(poses[0].data());
    
    ////////////////////////////////////////////////////////////////////////////
    /// for visualization
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::shared_ptr<open3d::geometry::LineSet>> camera_symbols_selected;
    for (int i = 0; i < poses.size(); i++) {
        if (!pose_added[i]) continue;
        std::vector<Eigen::Vector3d> symbol_points;
        std::vector<Eigen::Vector2i> lines;
        get_camera_symbol(poses[i], camera, symbol_points, lines);
        
        std::shared_ptr<open3d::geometry::LineSet> symbol{
            new open3d::geometry::LineSet(symbol_points, lines)};
        
        std::vector<Eigen::Vector3d> colors;
        for (int j = 0; j < 8; j++) colors.push_back(Eigen::Vector3d{0., 0., 1.});
        symbol->colors_ = colors;
        
        camera_symbols_selected.push_back(symbol);
    }
    
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.use_explicit_schur_complement = false;
    options.num_threads = 8;
    options.max_num_iterations = 10;

    ceres::Solver::Summary summary;
    
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    
    
    /* 
    How do we set the pyramid levels appropriately
    could we use pointers to pointers
    set a variable in each residual
    */
    
    
    // This is for setting the image pyramids to the full-size level
    
    for (int l = 0; l < pixels.size(); l++) {
        if (l % 5 != 0) continue;

        int i = 0;        
        for (int v = 0; v < visibilities[l].size(); v++) {
            if (visibilities[l][v] == source_views[l]) continue;
            residuals[i]->interpolator = &interpolator_pyramids[source_views[l]][0];
            residuals[i]->scale = 1.;
            i++;
        }
    }
    
    for (int l = 0; l < source_pixels.size(); l++) {
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                source_pixels[l].col(i)[0] = pixels[l][0] + x;
                source_pixels[l].col(i)[1] = pixels[l][1] + y;
                interpolator_pyramids[source_views[l]][0].Evaluate(source_pixels[l].col(i)[1], 
                                                                  source_pixels[l].col(i)[0],
                                                                  &source_patches[l][i]);
                i++;
            }
        }
        double mean = source_patches[l].sum() / 16;
        Eigen::Matrix<double, 16, 1> patch_centered{
            source_patches[l] - Eigen::Matrix<double, 16, 1>::Ones() *mean};
        source_patches[l] = patch_centered / patch_centered.norm();
    }
    std::cout << "Re-set parameters" << std::endl;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    
    
    
    RadialCamera<double> camera_optimized(camera_param);
    
    /* TODO: get color from pixel center */
    
    
    std::vector<Eigen::Vector3d> points_optimized;
    for (int i = 0; i < pixels.size(); i++) {
        if (i % 5 != 0) continue;
        Eigen::Vector3d X = unproject(pixels[i], planes[i], camera_optimized, poses[source_views[i]]);
        points_optimized.push_back(X);
    }
    std::vector<Eigen::Vector3d> points_combined;
    std::vector<Eigen::Vector3d> colors_combined;
    for (int i = 0; i < points_selected.size(); i++) {
        points_combined.push_back(points_selected[i]);
        colors_combined.push_back(Eigen::Vector3d(0., 0., 1.));
    }
    for (int i = 0; i < points_optimized.size(); i++) {
        points_combined.push_back(points_optimized[i]);
        colors_combined.push_back(Eigen::Vector3d(1., 0., 0.));
    }
    /* TODO commptue new normals */
    write_points_normals(output_path, points_optimized, normals);
    
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow();
    std::shared_ptr<open3d::geometry::PointCloud> cloud{
        new open3d::geometry::PointCloud(points_combined)};
    cloud->colors_ = colors_combined;
    vis.AddGeometry(cloud);
    
    std::vector<std::shared_ptr<open3d::geometry::LineSet>> camera_symbols_optimized;
    for (int i = 0; i < poses.size(); i++) {
        if (!pose_added[i]) continue;
        std::vector<Eigen::Vector3d> symbol_points;
        std::vector<Eigen::Vector2i> lines;
        get_camera_symbol(poses[i], camera, symbol_points, lines);
        
        std::shared_ptr<open3d::geometry::LineSet> symbol{
            new open3d::geometry::LineSet(symbol_points, lines)};
        
        std::vector<Eigen::Vector3d> colors;
        for (int j = 0; j < 8; j++) colors.push_back(Eigen::Vector3d{1., 0., 0.});
        symbol->colors_ = colors;
        
        camera_symbols_optimized.push_back(symbol);
    }
    
    for (int i = 0; i < camera_symbols_selected.size(); i++) {
        vis.AddGeometry(camera_symbols_selected[i]);
    }
    for (int i = 0; i < camera_symbols_optimized.size(); i++) {
        vis.AddGeometry(camera_symbols_optimized[i]);
    }
    
    vis.Run();
    
    /* TODO: store results somewhere */
}




