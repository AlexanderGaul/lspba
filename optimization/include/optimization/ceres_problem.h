#ifndef CERES_PROBLEM_H
#define CERES_PROBLEM_H

#include <vector>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include <optimization/problem_parameters.h>
#include <optimization/residuals.h>
#include <optimization/local_parameterization_se3.h>

void create_problem(
        ProblemParameters& parameters,
        std::vector<Eigen::Matrix<double, 2, 16>>& source_pixels,
        std::vector<Eigen::Matrix<double, 16, 1>>& source_patches,
        ceres::Problem& problem,
        int type, 
        int level=0, 
        bool landmarks_only=false,
        bool fix_pose_0=false, 
        bool fix_scale=false,
        std::vector<ceres::CostFunction*>* cost_functions=nullptr,
        std::vector<ceres::ResidualBlockId>* block_ids=nullptr
        ) {
    std::vector<bool> pose_added(parameters.poses.size(), false);
    
    for (long l = 0; l < parameters.pixels.size(); l++) {
        for (int v : parameters.visibilities[l]) {
            if (v == parameters.source_views[l]) continue;
            ceres::CostFunction* cost;
            
            switch (type) {
                case 0 :
                    cost = new ceres::AutoDiffCostFunction<
                            PatchResidual, 16,
                            3,
                            Sophus::SE3d::num_parameters, 
                            Sophus::SE3d::num_parameters, 
                            6>(new PatchResidual(source_patches[l].data(), 
                                                 source_pixels[l].data(),
                                                 &parameters.image_pyramids[v][level],
                                                 &parameters.basalt_pyramids[v][level]));
                    break;
                case 1 : {
                    ceres::NumericDiffOptions num_options;
                    num_options.relative_step_size = 1e-6;
                    num_options.ridders_relative_initial_step_size = 1e-8;
                    cost = new ceres::NumericDiffCostFunction<
                            PatchResidual, ceres::NumericDiffMethodType::CENTRAL, 16,
                            3,
                            Sophus::SE3d::num_parameters, 
                            Sophus::SE3d::num_parameters, 
                            6>(new PatchResidual(source_patches[l].data(), 
                                                 source_pixels[l].data(),
                                                 &parameters.image_pyramids[v][level],
                                                 &parameters.basalt_pyramids[v][level]), 
                               ceres::TAKE_OWNERSHIP, 16, num_options); }
                    break;
                case 2 :
                    cost = new PatchResidualStructureAnalytic(
                            source_pixels[l], source_patches[l], 
                            &parameters.image_pyramids[v][level],
                            &parameters.basalt_pyramids[v][level]);
                    break;
                default:
                    cost = new ceres::AutoDiffCostFunction<
                        PatchResidual, 16,
                        3,
                        Sophus::SE3d::num_parameters, 
                        Sophus::SE3d::num_parameters, 
                        6>(new PatchResidual(source_patches[l].data(), 
                                             source_pixels[l].data(),
                                             &parameters.image_pyramids[v][level],
                                             &parameters.basalt_pyramids[v][level]));
            }
            ceres::ResidualBlockId id;
            id = problem.AddResidualBlock(cost, new Rho(),
                                     parameters.planes[l].data(), 
                                     parameters.poses[parameters.source_views[l]].data(),
                                     parameters.poses[v].data(),
                                     parameters.camera_param.data());
            if (cost_functions) 
                cost_functions->push_back(cost);
            if (block_ids)
                block_ids->push_back(id);
            
            pose_added[v] = true;
            pose_added[parameters.source_views[l]] = true;
        }
    }
    for (int i = 0; i < parameters.poses.size(); i++) {
        if (pose_added[i]) {
            
            problem.AddParameterBlock(parameters.poses[i].data(), Sophus::SE3d::num_parameters,
                                  new Sophus::LocalParameterizationSE3);
            if (landmarks_only)
                problem.SetParameterBlockConstant(parameters.poses[i].data());
        }
        
    }
    if (false) {
        for (int i = 0; i < parameters.planes.size(); i++) {
            ceres::CostFunction* cost = 
                new ceres::AutoDiffCostFunction<
                    AngleCostFunctor,
                    1, 3>(new AngleCostFunctor(parameters.get_normal(i)));
            ceres::ResidualBlockId id = problem.AddResidualBlock(cost, nullptr, parameters.planes[i].data());
        }
    }
    if (fix_scale) {
        double min_distance = (parameters.poses[0].translation() - parameters.poses[0].translation()).norm();
        int i_min = 0;
        int j_min = 1;
        for (int i = 0; i < parameters.poses.size(); i++) {
            if (!pose_added[i]) continue;
            for (int j = i; j < parameters.poses.size(); j++) {
                if (!pose_added[j]) continue;
                double distance = (parameters.poses[0].translation() - parameters.poses[0].translation()).norm();
                if (distance < min_distance) {
                    min_distance = distance;
                    i_min = i;
                    j_min = j;
                }
            }
        }
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<
                DistanceCostFunctor, 1,
                Sophus::SE3d::num_parameters, 
                Sophus::SE3d::num_parameters>(new DistanceCostFunctor(min_distance));
        ceres::ResidualBlockId id = problem.AddResidualBlock(cost, nullptr, 
                                                             parameters.poses[i_min].data(),
                                                             parameters.poses[j_min].data());
        
        if (cost_functions) cost_functions->push_back(cost);
        if (block_ids) block_ids->push_back(id);
    }
    if (fix_pose_0)
        problem.SetParameterBlockConstant(parameters.poses[0].data());
    if (landmarks_only)
        problem.SetParameterBlockConstant(parameters.camera_param.data());
    
}




// TODO: options as arguments
void solve_problem(ProblemParameters& data,
                   int level, int type, int iterations) {
    
    double scale = pow(2, level);
    
    
    // TODO use compute patches here
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches;
    compute_patches(data, level, source_pixels, source_patches);
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels_dup;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches_dup;
    for (int i = 0; i < source_pixels.size(); i++)
        source_pixels_dup.push_back(source_pixels[i]);
    for (int i = 0; i < source_pixels.size(); i++)
        source_patches_dup.push_back(source_patches[i]);
    std::vector<bool> pose_added;
    for (int i = 0; i < data.poses.size(); i++) {
        pose_added.push_back(false);
    }
    
    data.camera_param.head<4>() /= scale;    
    
    ceres::Problem problem;
    create_problem(data, source_pixels, source_patches, problem, type, level, 
                   false, true, true, nullptr, nullptr);

    std::cout << "Created problem" << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.use_explicit_schur_complement = false;
    //options.initial_trust_region_radius = 1e-2;
    options.num_threads = 8;
    options.max_num_iterations = iterations;
    
    //options.check_gradients = true;
    //options.gradient_check_relative_precision = 1e-8;
    //options.gradient_check_numeric_derivative_relative_step_size = 1e-5;

    ceres::Solver::Summary summary;
    
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    
    data.camera_param.head<4>() *= scale;
}

#endif // CERES_PROBLEM_H
