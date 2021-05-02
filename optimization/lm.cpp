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

#include <optimization/ceres_problem.h>

#include <optimization/io.h>
//#include <optimization/camera.h>
#include <optimization/residuals.h>
#include <optimization/local_parameterization_se3.h>
//#include <optimization/utils.h>
#include <optimization/patchplt.h>
#include <optimization/evaluation.h>

#include <optimization/test_jacobians.h>

#include <basalt/image/image.h>



int main(int argc, char **argv) {
    // TODO: change into filesystem paths
    // TODO: create outptu folder
    // TODO: create output name from settings
    std::string workspace_path = "../../../data/training/ignatius_subset_workspace/";
    //std::string workspace_path = "../../../data/horse_workspace/";
    //workspace_path = "../../../data/rendering/family_image_workspace/";
    //workspace_path = "/home/alexander/Documents/studies/20_ws/idp/data/rendering/family_image_workspace_gt";
    std::string preprocessing_folder = "preprocessing_low/";
    std::string preprocessing_path;
    std::string output_path;
    int modulo_selection = 1;
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
        output_path = workspace_path + "/optimization/" + "preprocessing_low_1010_perturbed/"; //preprocessing_folder; // "preprocessing_rendered/";//preprocessing_folder;
    }
    if (argc > 4) {
        modulo_selection = atoi(argv[4]);
    }
    
    // TODO: create output folder
    
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
    /*
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
    } */
    
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




