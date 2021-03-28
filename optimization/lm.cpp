#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

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
    
    std::vector<std::array<cv::Mat, 3>> mat_pyramids;
    std::vector<std::vector<ceres::Grid2D<double, 1>>> grid_pyramids;
    std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>> interpolator_pyramids;
    /* TODO: reserve space in vectors */
    // Read images
    for (int i = 0; i < poses.size(); i++) {
        cv::Mat mat1;
        cv::Mat mat2;
        std::string im_file = image_path + (std::string((6 - std::to_string(i+1).length()), '0') + std::to_string(i+1) + ".jpg");
        
        images_color.push_back(cv::imread(im_file));
        
        // Create image pyramids with OpenCV
        cv::imread(im_file, cv::IMREAD_GRAYSCALE).convertTo(mat1, CV_64F);
        // TODO: test this
        if (true) {
            cv::pyrDown(mat1, mat2); //cv::BORDER_REPLICATE
        }
        mat1 /= 255.;
        mat2 /= 255.;
        mat_pyramids.push_back({mat1, mat2});
    }
    for (int i = 0; i < poses.size(); i++) {
        grid_pyramids.push_back({ceres::Grid2D<double, 1>((double*)(mat_pyramids[i][0].data), 0, mat_pyramids[i][0].rows, 0, mat_pyramids[i][0].cols),
                                 ceres::Grid2D<double, 1>((double*)(mat_pyramids[i][1].data), 0, mat_pyramids[i][1].rows, 0, mat_pyramids[i][1].cols)});
    }
    for (int i = 0; i < poses.size(); i++) {
        interpolator_pyramids.push_back({ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids[i][0]),
                                         ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids[i][1])});
    }
    
    problem_parameters param_all(planes, poses, camera_param, pixels, source_views, visibilities, interpolator_pyramids);
    
    problem_parameters param_initial(param_all, selection);
    problem_parameters param_initial_perturbed{param_initial};
    
    RadialCamera<double> camera{camera_param};
    
    std::vector<Eigen::Vector3d> p3D = param_initial_perturbed.get_points();
    std::vector<Eigen::Vector3d> normals_init = param_initial_perturbed.get_normals();
    std::vector<Eigen::Vector3d> planes_perturbed;
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.005);
    for (int i = 0; i < p3D.size(); i++) {
        Eigen::Vector3d p_C = poses[param_initial_perturbed.source_views[i]] * p3D[i];
        double perturbation = -100.;
        while (abs(perturbation) > 0.01) {
            perturbation = distribution(generator);
        }
        p_C[2] += perturbation;
        planes_perturbed.push_back(
                    get_plane(p_C, 
                              poses[param_initial_perturbed.source_views[i]].rotationMatrix() * normals_init[i]));
    }
    param_initial_perturbed.planes = planes_perturbed;
    
    problem_parameters param_optimize{param_initial};
    problem_parameters param_optimize_perturbed{param_initial_perturbed};

    solve_problem(param_optimize, 1);
    solve_problem(param_optimize, 0);
    
    solve_problem(param_optimize_perturbed, 1);
    solve_problem(param_optimize_perturbed, 0);
    
    std::vector<Eigen::Vector3d> ps;
    std::vector<Eigen::Vector3d> ns;
    write_points_normals(output_path+"points_normals_initial.txt", 
                         ps=param_initial.get_points(), ns=param_initial.get_normals());
    write_points_normals(output_path+"points_normals_perturbed_initial.txt", 
                         ps=param_initial_perturbed.get_points(), ns=param_initial_perturbed.get_normals());
    write_points_normals(output_path+"points_normals_optimized.txt", 
                         ps=param_optimize.get_points(), ns=param_optimize.get_normals());
    write_points_normals(output_path+"points_normals_perturbed_optimized.txt", 
                         ps=param_optimize_perturbed.get_points(), ns=param_optimize_perturbed.get_normals());
}




