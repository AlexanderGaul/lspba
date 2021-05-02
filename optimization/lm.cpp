#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <filesystem>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
//#include <open3d/Open3D.h>

#include <sophus/se3.hpp>

#include <optimization/loader.h>
#include <optimization/io.h>
#include <optimization/camera.h>
#include <optimization/residuals.h>
#include <optimization/local_parameterization_se3.h>
#include <optimization/utils.h>

//#include <basalt/image/image.h>

class problem_parameters {
    public:
    //problem_parameters() = default;
    
    problem_parameters(std::vector<Eigen::Vector3d> planes,
                       std::vector<Sophus::SE3d> poses,
                       RadialCamera<double>::VecN camera_param,
                       std::vector<Eigen::Vector2d> pixels,
                       std::vector<int> source_views,
                       std::vector<std::vector<int>> visibilities,
                       std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>>& image_pyramids) :
        planes(planes), poses(poses), camera_param(camera_param), 
        pixels(pixels), source_views(source_views), visibilities(visibilities),
        image_pyramids(image_pyramids) {}
    
    // TODO: constructor based on
    problem_parameters(problem_parameters& param, std::vector<long> selection) :
        poses(param.poses), camera_param(param.camera_param), image_pyramids(param.image_pyramids) {
        for (long i : selection) {
            planes.push_back(param.planes[i]);
            pixels.push_back(param.pixels[i]);
            source_views.push_back(param.source_views[i]);
            visibilities.push_back(param.visibilities[i]);
        }
    }
    
    std::vector<Eigen::Vector3d> planes;
    std::vector<Sophus::SE3d> poses;
    RadialCamera<double>::VecN camera_param;
    
    // TODO: should we turn this into references since they don't change??
    // TOOD: is there a use case for changing these
    std::vector<Eigen::Vector2d> pixels;
    std::vector<int> source_views;
    std::vector<std::vector<int>> visibilities;
    std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>>& image_pyramids;
    
    // TODO: return vector or accept vector 
    std::vector<Eigen::Vector3d> get_normals() {
        std::vector<Eigen::Vector3d> normals;
        RadialCamera<double> camera{camera_param};
        for (int i = 0; i < pixels.size(); i++) {
            normals.push_back(get_normal(unproject(pixels[i], planes[i], camera, poses[source_views[i]]), 
                                         planes[i], poses[source_views[i]]));
        }
        return normals;
    }
    
    std::vector<Eigen::Vector3d> get_points() {
        std::vector<Eigen::Vector3d> points;
        RadialCamera<double> camera{camera_param};
        for (int i = 0; i < pixels.size(); i++) {
            points.push_back(unproject(pixels[i], planes[i], camera, poses[source_views[i]]));
        }
        return points;
    }
};

void compute_patches(problem_parameters& data,
                     int level,
                     std::vector<Eigen::Matrix<double, 2, 16>>& source_pixels,
                     std::vector<Eigen::Matrix<double, 16, 1>>& source_patches
                     ) {
    double scale = pow(2, level);
    for (int l = 0; l < data.pixels.size(); l++) {
        Eigen::Matrix<double, 2, 16> grid;
        Eigen::Matrix<double, 16, 1> patch;
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                grid.col(i)[0] = data.pixels[l][0] + scale * x;
                grid.col(i)[1] = data.pixels[l][1] + scale * y;
                data.image_pyramids[data.source_views[l]][level].Evaluate((grid.col(i)[1]) / scale, 
                                                                          (grid.col(i)[0]) / scale,
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
}

void solve_problem(problem_parameters& data,
                   int level) {
    
    double scale = pow(2, level);
    // TODO: use some eigen here
    data.camera_param[0] /= scale;
    data.camera_param[1] /= scale;
    data.camera_param[2] /= scale;
    data.camera_param[3] /= scale;
    
    
    // TODO use compute patches here
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches;
    compute_patches(data, level, source_pixels, source_patches);
    
    std::vector<bool> pose_added;
    for (int i = 0; i < data.poses.size(); i++) {
        pose_added.push_back(false);
    }
    ceres::Problem problem;
    for (long l = 0; l < data.pixels.size(); l++) {
        for (int v : data.visibilities[l]) {
            if (v == data.source_views[l]) continue;
            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<PatchResidual, 16,
                                                Sophus::SE3d::num_parameters, 
                                                Sophus::SE3d::num_parameters, 
                                                6, 
                                                3>(new PatchResidual(source_patches[l].data(), 
                                                                     source_pixels[l].data(),
                                                                     &data.image_pyramids[data.source_views[l]][level]));
            problem.AddResidualBlock(cost, new Rho(), data.poses[data.source_views[l]].data(),
                                                      data.poses[v].data(),
                                                      data.camera_param.data(),
                                                      data.planes[l].data());
            pose_added[v] = true;
            pose_added[data.source_views[l]] = true;
            break;
        }
    }
    // TODO: automatically determine largest distance
    //ceres::CostFunction* cost = 
    //    new ceres::AutoDiffCostFunction<DistanceCostFunctor, 1, 
    //                                    Sophus::SE3d::num_parameters, 
    //                                    Sophus::SE3d::num_parameters>(new DistanceCostFunctor((data.poses[58].translation() - 
    //                                                                                           data.poses[85].translation()).norm()));
    //problem.AddResidualBlock(cost, nullptr, data.poses[58].data(), data.poses[85].data());
    
    for (int i = 0; i < data.poses.size(); i++) {
        if (pose_added[i]) {
            problem.AddParameterBlock(data.poses[i].data(), Sophus::SE3d::num_parameters,
                                  new Sophus::LocalParameterizationSE3);
            problem.SetParameterBlockConstant(data.poses[i].data());
        }
        
    }
    //problem.SetParameterBlockConstant(poses[0].data());
    problem.SetParameterBlockConstant(data.camera_param.data());
    
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.use_explicit_schur_complement = false;
    options.num_threads = 8;
    options.max_num_iterations = 10;
    
    //options.check_gradients = true;
    //options.gradient_check_relative_precision = 1e-2;

    ceres::Solver::Summary summary;
    
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    
    // TODO: use some eigen here
    data.camera_param[0] *= scale;
    data.camera_param[1] *= scale;
    data.camera_param[2] *= scale;
    data.camera_param[3] *= scale;
}

int main(int argc, char **argv) {
    std::string workspace_path = "../../../data/training/ignatius_subset_workspace/";
    std::string preprocessing_folder = "preprocessing_low/";
    std::string preprocessing_path;
    std::string output_path;
    int modulo_selection = 5;
    if (argc > 1) {
        workspace_path = argv[1];
    } 
    if (argc > 2) {
        preprocessing_folder = argv[2];
    }
    preprocessing_path = workspace_path + "/" + preprocessing_folder;
    if (argc > 3) {
        output_path = argv[3];
    } else {
        output_path = workspace_path + "/optimization/" + preprocessing_folder;
    }
    if (argc > 4) {
        modulo_selection = atoi(argv[4]);
    }

    std::string camera_path{preprocessing_path + "/camera.txt"};
    std::string pose_path{preprocessing_path + "/poses.txt"};
    std::string image_path{workspace_path + "/images/"};
    std::string landmark_path{preprocessing_path + "/landmarks.txt"};
    std::string source_path{preprocessing_path + "/source_views.txt"};
    std::string visibility_path{preprocessing_path + "/visibility.txt"};
    std::string points3d_path{preprocessing_path + "/points_normals_gridscales_select.txt"};

    
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
    std::vector<long> selection;
    for (long i = 0; i < pixels.size(); i++) {
        if (i % modulo_selection == 0)
            selection.push_back(i);
    }

    std::cout << "Landmarks: " << pixels.size() << std::endl;
    std::cout << "Observations: " << observations << std::endl;
    std::cout << "Selection: " << selection.size() << std::endl;
    
    std::vector<cv::Mat> images_color;
    load_images(image_path, images_color);
    
    std::vector<std::vector<cv::Mat>> mat_pyramids;
    std::vector<std::vector<ceres::Grid2D<double, 1>>> grid_pyramids;
    std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>> interpolator_pyramids;
    std::vector<std::vector<basalt::Image<double>>> basalt_pyramids;
    
    create_pyramid_images(images_color,
                          mat_pyramids,
                          &grid_pyramids,
                          &interpolator_pyramids,
                          &basalt_pyramids);
    
    
    ProblemParameters param_all(planes, poses, camera_param, pixels, 
                                 source_views, visibilities, interpolator_pyramids, basalt_pyramids,
                                 images_color, mat_pyramids);
    //param_all.filter_extreme_angles();
    
    ProblemParameters param_initial(param_all, selection);
    
    double max_x = 0.;
    double max_y = 0.;
    for (int i = 0; i < param_initial.planes.size(); i++) {
        if (param_initial.get_point(i)[0] > max_x) max_x = param_initial.get_point(i)[0];
        if (param_initial.get_point(i)[1] > max_y) max_y = param_initial.get_point(i)[1];
    }
    std::cout << "-------" << std::endl;
    std::cout << max_x << std::endl;
    std::cout << max_y << std::endl;
    
    std::vector<long> selection_center;
    for (int i = 0; i < param_all.pixels.size(); i++) {
        Eigen::Vector3d p = param_all.get_point(i);
        if (p[0] < 0.075 && p[0] > -0.075 && p[1] < 0.075 && p[1] > -0.075)
            selection_center.push_back(i);
    }
    
    //param_initial.filter_extreme_angles();
    ProblemParameters param_center(param_all, selection_center);
    ProblemParameters param_initial_perturbed = perturb_data(param_initial);
    

    
    std::cout << param_initial.get_normal(0).transpose() << std::endl;
    std::cout << param_initial_perturbed.get_normal(0).transpose() << std::endl;
    std::cout << "Perturbation comparison" << std::endl;
    std::cout << param_initial_perturbed.get_normals()[0].transpose() << std::endl;
    std::cout << z_error(param_initial, -0.25) << std::endl;
    std::cout << z_error(param_initial_perturbed, -0.25) << std::endl;
    std::cout << normal_error(param_initial, {0., 0., 1.}) << std::endl;
    std::cout << normal_error(param_initial_perturbed, {0., 0., 1.}) << std::endl;
    
    //std::cout << "Testing jacobians" << std::endl;
    //test_problem(param_optimize, 1);
    
    std::cout << "Solving Problems..." << std::endl;
    //test_problem(param_optimize, 0, 2);
    
    ProblemParameters param_optimize_lvl{param_initial};
    solve_problem(param_optimize_lvl, 1, 2, 10);
    ProblemParameters param_optimize{param_optimize_lvl};
    solve_problem(param_optimize, 0, 2, 10);
    
    ProblemParameters param_optimize_single{param_initial};
    
    //solve_problem(param_optimize_single, 0, 2, 20);
    /*
    plt::figure(1);
    plot_patches(param_initial, 500);
    plt::figure(2);
    plot_patches(param_optimize, 500);
    plt::show();
    */
    std::cout << "Solving perturbed problem... " << std::endl;
    ProblemParameters param_optimize_perturbed_lvl{param_initial_perturbed};
    //solve_problem(param_optimize_perturbed_lvl, 1, 2, 10);
    ProblemParameters param_optimize_perturbed{param_optimize_perturbed_lvl};
    //solve_problem(param_optimize_perturbed, 0, 2, 10);
    
    ProblemParameters param_optimize_perturbed_single{param_initial_perturbed};
    //solve_problem(param_optimize_perturbed_single, 0, 2, 20);
    
    std::vector<double> bins_z{0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005};
    std::vector<long> histogram_z_initial = z_error_histogram(param_initial, -0.25, bins_z);
    std::vector<long> histogram_z_optimize = z_error_histogram(param_optimize, -0.25, bins_z);
    std::vector<long> histogram_z_optimize_fine = z_error_histogram(param_optimize_single, -0.25, bins_z);
    print_std_vector(histogram_z_initial);
    print_std_vector(histogram_z_optimize);
    print_std_vector(histogram_z_optimize_fine);
    
    std::vector<double> bins_n{0.001, 0.01, 0.1};
    std::vector<long> histogram_n_initial = normal_error_histogram(param_initial, {0., 0., 1.}, bins_n);
    std::vector<long> histogram_n_optimize = normal_error_histogram(param_optimize, {0., 0., 1.}, bins_n);
    std::vector<long> histogram_n_optimize_fine = normal_error_histogram(param_optimize_single, {0., 0., 1.}, bins_n);
    print_std_vector(histogram_n_initial);
    print_std_vector(histogram_n_optimize);
    print_std_vector(histogram_n_optimize_fine);
    
    std::vector<long> histogram_n_perturbed = normal_error_histogram(param_initial_perturbed, {0., 0., 1.}, bins_n);
    std::vector<long> histogram_n_perturbed_opt = normal_error_histogram(param_optimize_perturbed, {0., 0., 1.}, bins_n);
    print_std_vector(histogram_n_perturbed);
    print_std_vector(histogram_n_perturbed_opt);
    
    //std::cout << param_initial.get_points()[0].transpose() << std::endl;
    //std::cout << param_initial.get_normals()[0].transpose() << std::endl;
    for (int i = 0; i < 200; i++) {
        if (i % 100 != 0) continue;
        //if (normal_error(param_optimize_perturbed.get_normal(i), {0., 0., 1.}) < 0.1) continue;
        //if ((param_optimize.get_point(i) - param_initial.get_point(i)).norm() < 0.005) continue;
        plt::figure(1);
        plot_patches(param_initial, i);
        //plt::figure(2);
        //plot_patches(param_optimize_lvl, i);
        plt::figure(3);
        plot_patches(param_optimize, i);
        plt::figure(4);
        plot_patches(param_initial_perturbed, i);
        //plt::figure(5);
        //plot_patches(param_optimize_perturbed_lvl, i);
        plt::figure(6);
        plot_patches(param_optimize_perturbed, i);
        plt::show();
    }
    
    std::vector<Eigen::Vector3d> ps;
    std::vector<Eigen::Vector3d> ns;
    write_points_normals(output_path+"points_normals_complete.txt", 
                         ps=param_all.get_points(), ns=param_all.get_normals());
    
    write_points_normals(output_path+"points_normals_initial.txt", 
                         ps=param_initial.get_points(), ns=param_initial.get_normals());
    std::cout << param_initial.poses[param_initial.source_views[0]] * ps[0] << std::endl;
    std::cout << unproject_camera(param_initial.pixels[0], param_initial.planes[0], RadialCamera<double>(param_initial.camera_param)) << std::endl;
    
    write_points_normals(output_path+"points_normals_perturbed_initial.txt", 
                         ps=param_initial_perturbed.get_points(), ns=param_initial_perturbed.get_normals());
    std::cout << param_initial_perturbed.poses[param_initial_perturbed.source_views[0]] * ps[0] << std::endl;
    std::cout << unproject_camera(param_initial_perturbed.pixels[0], param_initial_perturbed.planes[0], 
                                  RadialCamera<double>(param_initial_perturbed.camera_param)) << std::endl;
    
    write_points_normals(output_path+"points_normals_optimized.txt", 
                         ps=param_optimize.get_points(), ns=param_optimize.get_normals());
    std::cout << param_optimize.poses[param_optimize.source_views[0]] * ps[0] << std::endl;
    std::cout << unproject_camera(param_optimize.pixels[0], param_optimize.planes[0], RadialCamera<double>(param_optimize.camera_param)) << std::endl;
    
    write_points_normals(output_path+"points_normals_perturbed_optimized.txt", 
                         ps=param_optimize_perturbed.get_points(), ns=param_optimize_perturbed.get_normals());
    std::cout << param_optimize_perturbed.poses[param_optimize_perturbed.source_views[0]] * ps[0] << std::endl;
    std::cout << unproject_camera(param_optimize_perturbed.pixels[0], param_optimize_perturbed.planes[0], 
                                  RadialCamera<double>(param_initial.camera_param)) << std::endl;
                         
}




