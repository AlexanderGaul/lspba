#ifndef TEST_JACOBIANS_H
#define TEST_JACOBIANS_H

#include <vector>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include <optimization/camera.h>
#include <optimization/problem_parameters.h>
#include <optimization/residuals.h>
#include <optimization/local_parameterization_se3.h>
#include <optimization/ceres_problem.h>


struct PatchResidualStructureAnalyticFunctor {
    PatchResidualStructureAnalyticFunctor(
          const Eigen::Matrix<double, 2, 16>& source_pixels,
          const Eigen::Matrix<double, 16, 1>& source_patch,
          const Sophus::SE3d& source_pose,
          const Sophus::SE3d& target_pose,
          const RadialCamera<double>::VecN& camera_param,
          ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator,
          basalt::Image<double>* basalt_image, 
          long landmark_index=-1, int visibility_index=-1,
          ProblemParameters* parameters=nullptr) :
        residual(source_pixels, source_patch,
                 interpolator, basalt_image,
                 landmark_index, visibility_index, parameters) {}
    
    bool operator() (const double* const parameter,
                     double* sresidual) const {
        const double* const par = parameter;
        return residual.Evaluate(&par, sresidual, nullptr);
    }
    
    PatchResidualStructureAnalytic residual;
};

struct NCCFunctor {  
    bool operator() (const double* const parameter,
                     double* sresidual) const {
        Eigen::Map<Eigen::Matrix<double, 16, 1> const> patch{parameter};
        Eigen::Map<Eigen::Matrix<double, 16, 1>> patch_ncc{sresidual};
        double mu = patch.mean();
        Eigen::Matrix<double, 16, 1> patch_centered = 
            patch - Eigen::Matrix<double, 16, 1>::Ones() * mu;
        patch_ncc = patch_centered.normalized();
        return true;
    }
};
class NCCCost : public ceres::SizedCostFunction<16, 16> {
public:
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<Eigen::Matrix<double, 16, 1> const> patch{parameters[0]};
        Eigen::Map<Eigen::Matrix<double, 16, 1>> patch_ncc{residuals};
        Eigen::Map<Eigen::Matrix<double, 16, 16>> jacobian{jacobians[0]};
        Eigen::Matrix<double, 16, 16> jac;
        Eigen::Matrix<double, 16, 1> n;
        ncc(patch, n, &jac);
        patch_ncc = n;
        jacobian = jac;
        return true;
    }
};

class ProjectionCost : public ceres::SizedCostFunction<2, RadialCamera<double>::N> {
public:
    ProjectionCost(Eigen::Vector3d vec) : vec(vec) {}
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<Eigen::Matrix<double, 6, 1> const> const intrinsics(parameters[0]);
        std::shared_ptr<RadialCamera<double>> camera(new RadialCamera<double>(intrinsics));
        Eigen::Map<Eigen::Vector2d> res{residuals};
        Eigen::Matrix<double, 2, RadialCamera<double>::N> jacobian;
        
        res = camera->project(vec, nullptr, &jacobian);
        
        if (jacobians && jacobians[0]) { 
            Eigen::Map<Eigen::Matrix<double, RadialCamera<double>::N, 2>> jacobian_out{jacobians[0]};
            jacobian_out = jacobian.transpose();
        }
        return true;
    }
    
    Eigen::Vector3d vec;
};
struct ProjectionFunctor {
    ProjectionFunctor(Eigen::Vector3d vec) : vec(vec) {}
    
    template<typename T>
    bool operator() (T const * const parameter, T* residual) const {
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(parameter);
        std::shared_ptr<RadialCamera<T>> camera(new RadialCamera<T>(intrinsics));
        Eigen::Map<Eigen::Matrix<T, 2, 1>> res{residual};
        
        res = camera->project({T(vec[0]), T(vec[1]), T(vec[2])});
        return true;
    }
    
    Eigen::Vector3d vec;
};

class UnprojectionCost : public ceres::SizedCostFunction<3, RadialCamera<double>::N> {
  public:
    UnprojectionCost(Eigen::Vector2d vec) : vec(vec) {}
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<RadialCamera<double>::VecN const> param{parameters[0]};
        std::shared_ptr<RadialCamera<double>> camera(new RadialCamera<double>(param));
        Eigen::Map<Eigen::Matrix<double, 3, 1>> res{residuals};
        Eigen::Matrix<double, 3, RadialCamera<double>::N> jacobian;
        
        res = camera->unproject(vec, &jacobian);
        
        if (jacobians) {
            Eigen::Map<Eigen::Matrix<double, RadialCamera<double>::N, 3>> jacobian_out{jacobians[0]};
            jacobian_out = jacobian.transpose();
        }
        return true;
    }
    
    Eigen::Vector2d vec;
};
struct UnprojectionFunctor {
    UnprojectionFunctor(Eigen::Vector2d vec) : vec(vec) {}
    
    template<typename T>
    bool operator() (T const * const parameter, T* residual) const {
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(parameter);
        std::shared_ptr<RadialCamera<T>> camera(new RadialCamera<T>(intrinsics));
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res{residual};
        
        res = camera->unproject({T(vec[0]), T(vec[1])});
        return true;
    }
    
    Eigen::Vector2d vec;
};
class CorrespondenceCost : public ceres::SizedCostFunction<32, RadialCamera<double>::N> {
public:
    CorrespondenceCost(Eigen::Matrix<double, 2, 16> pixel,
                       Eigen::Vector3d plane,
                       Sophus::SE3d source_pose,
                       Sophus::SE3d target_pose) : 
        pixel(pixel), plane(plane), source_pose(source_pose), target_pose(target_pose) {}
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<RadialCamera<double>::VecN const> param{parameters[0]};
        std::shared_ptr<RadialCamera<double>> camera(new RadialCamera<double>(param));
        Eigen::Map<Eigen::Matrix<double, 2, 16>> res{residuals};
        std::vector<Eigen::Matrix<double, 2, RadialCamera<double>::N>> jacobian(16);
        
        Eigen::Matrix<double, 2, 16> target_pixels;
        correspondence(pixel, plane, source_pose, target_pose, *camera, target_pixels, 
                       nullptr, nullptr, nullptr, &jacobian[0]);
        res = target_pixels;
        
        if (jacobians) {
            Eigen::Map<Eigen::Matrix<double, RadialCamera<double>::N, 32>> jacobian_out{jacobians[0]};
            for (int i = 0; i < 16; i++) {
                jacobian_out.block<RadialCamera<double>::N, 2>(0, 2*i) = jacobian[i].transpose();
            }
        }
        return true;
    }
    
    Eigen::Matrix<double, 2, 16> pixel;
    Eigen::Vector3d plane;
    Sophus::SE3d source_pose;
    Sophus::SE3d target_pose;
};

struct CorrespondenceFunctor {
    CorrespondenceFunctor(Eigen::Matrix<double, 2, 16> pixel,
                       Eigen::Vector3d plane,
                       Sophus::SE3d source_pose,
                       Sophus::SE3d target_pose) : 
        pixel(pixel), plane(plane), source_pose(source_pose), target_pose(target_pose) {}
    
    bool operator() (double const * const parameter, double* residual) const {
        Eigen::Map<RadialCamera<double>::VecN const> param{parameter};
        std::shared_ptr<RadialCamera<double>> camera(new RadialCamera<double>(param));
        Eigen::Map<Eigen::Matrix<double, 2, 16>> res{residual};
        std::vector<Eigen::Matrix<double, 2, RadialCamera<double>::N>> jacobian(16);
        
        Eigen::Matrix<double, 2, 16> target_pixels;
        correspondence(pixel, plane, source_pose, target_pose, *camera, target_pixels, 
                       nullptr, nullptr, nullptr, nullptr);
        res = target_pixels;
        
        return true;
    }
    
    Eigen::Matrix<double, 2, 16> pixel;
    Eigen::Vector3d plane;
    Sophus::SE3d source_pose;
    Sophus::SE3d target_pose;
};



class RotationCost : public ceres::SizedCostFunction<3, Sophus::SE3d::num_parameters> {
public:
    RotationCost(Eigen::Vector3d vec) : vec(vec) {}
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<Sophus::SE3d const> pose{parameters[0]};
        Eigen::Map<Eigen::Vector3d> res{residuals};
        
        res = pose * vec;
        if (jacobians) {
            Eigen::Map<Eigen::Matrix<double, Sophus::SE3d::num_parameters, 3>> jac{jacobians[0]};
            jac.block<4, 3>(0, 0) = quaternion_vector_derivative<double>(pose.unit_quaternion(), vec).transpose();
            jac.block<3, 3>(4, 0) = Eigen::Matrix<double, 3, 3>::Identity();
        }
        
        return true;
    }
    
    Eigen::Vector3d vec;
};
struct RotationFunctor {
    RotationFunctor(Eigen::Vector3d vec) : vec(vec) {}
    
    template<typename T>
    bool operator() (T const * const parameter, T* residual) const {
        Eigen::Map<Sophus::SE3<T> const> pose{parameter};
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res{residual};
        Eigen::Matrix<T, 3, 1> vec_T{T(vec[0]), T(vec[1]), T(vec[2])};
        res = pose * vec_T;
        return true;
    }
    
    Eigen::Vector3d vec;
};


void create_analytic_as_numeric(
        ProblemParameters& parameters,
        std::vector<Eigen::Matrix<double, 2, 16>>& source_pixels,
        std::vector<Eigen::Matrix<double, 16, 1>>& source_patches,
        ceres::Problem& problem,
        int type, 
        int level=0, 
        bool fix_pose_0=true, 
        bool landmarks_only=false,
        std::vector<ceres::CostFunction*>* cost_functions=nullptr,
        std::vector<ceres::ResidualBlockId>* block_ids=nullptr
        ) {
    std::vector<bool> pose_added(parameters.poses.size(), false);
    for (long l = 0; l < parameters.pixels.size(); l++) {
        for (int v : parameters.visibilities[l]) {
            if (v == parameters.source_views[l]) continue;
            ceres::CostFunction* cost;
            
            cost = new ceres::NumericDiffCostFunction<
                    PatchResidualStructureAnalyticFunctor, ceres::NumericDiffMethodType::CENTRAL, 16,
                    3>(new PatchResidualStructureAnalyticFunctor(
                           source_pixels[l], source_patches[l], 
                           parameters.poses[parameters.source_views[l]], parameters.poses[v], parameters.camera_param,
                           &parameters.image_pyramids[v][level],
                           &parameters.basalt_pyramids[v][level]));

            ceres::ResidualBlockId id;
            id = problem.AddResidualBlock(cost, 
                                     new Rho(), 
                                     parameters.planes[l].data());
            if (cost_functions) 
                cost_functions->push_back(cost);
            if (block_ids)
                block_ids->push_back(id);
            
            pose_added[v] = true;
            pose_added[parameters.source_views[l]] = true;
        }
    }
    if (type != 2) {
        for (int i = 0; i < parameters.poses.size(); i++) {
            if (pose_added[i]) {
                problem.AddParameterBlock(parameters.poses[i].data(), Sophus::SE3d::num_parameters,
                                      new Sophus::LocalParameterizationSE3);
                if (landmarks_only)
                    problem.SetParameterBlockConstant(parameters.poses[i].data());
            }
            
        }
        if (fix_pose_0)
            problem.SetParameterBlockConstant(parameters.poses[0].data());
        if (landmarks_only)
            problem.SetParameterBlockConstant(parameters.camera_param.data());
    }
}


void test_problem(ProblemParameters& data, int level=0) {
    double scale = pow(2, level);
    
    
    // TODO use compute patches here
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches;
    compute_patches(data, level, source_pixels, source_patches);
    std::vector<Eigen::Matrix<double, 2, 16>> source_pixels_dup;
    std::vector<Eigen::Matrix<double, 16, 1>> source_patches_dup;
    compute_patches(data, level, source_pixels_dup, source_patches_dup, false);
    std::vector<bool> pose_added;
    for (int i = 0; i < data.poses.size(); i++) {
        pose_added.push_back(false);
    }
    
    data.camera_param.head<4>() /= scale;
    
    
    
    
    

    ceres::Problem problem_automatic;
    std::vector<ceres::CostFunction*> costs_automatic;
    std::vector<ceres::ResidualBlockId> ids_automatic;
    create_problem(data, source_pixels, source_patches, problem_automatic, 
                   0, level, false, false, false, &costs_automatic, &ids_automatic);
    
    ceres::Problem problem_numeric;
    std::vector<ceres::CostFunction*> costs_numeric;
    std::vector<ceres::ResidualBlockId> ids_numeric;
    create_problem(data, source_pixels, source_patches, problem_numeric, 
                   1, level, false, false, false, &costs_numeric, &ids_numeric);
    
    ceres::Problem problem_analytic;
    std::vector<ceres::CostFunction*> costs_analytic;
    std::vector<ceres::ResidualBlockId> ids_analytic;
    create_problem(data, source_pixels, source_patches, problem_analytic, 
                   2, level, false, false, false, &costs_analytic, &ids_analytic);
    
    ceres::Problem problem_analytic_num;
    std::vector<ceres::CostFunction*> costs_analytic_num;
    std::vector<ceres::ResidualBlockId> ids_analytic_num;
    create_analytic_as_numeric(
                   data, source_pixels, source_patches, problem_analytic_num, 
                   2, level, false, false, &costs_analytic_num, &ids_analytic_num);
    
    ceres::Problem::EvaluateOptions test_options;
    
    
    
    for (int i = 0; i < 4; i++) {
        double cost;
        Eigen::Matrix<double, 16, 1> residuals;
        Eigen::Matrix<double, 16, 3> jacobian_plane;
        Eigen::Matrix<double, Sophus::SE3d::num_parameters, 16> jacobian_source_pose;
        Eigen::Matrix<double, Sophus::SE3d::num_parameters, 16> jacobian_target_pose;
        Eigen::Matrix<double, 16, RadialCamera<double>::N> jacobian_intrinsics;
        std::vector<double*> jac_ptrs;
        jac_ptrs.push_back(jacobian_plane.data());
        jac_ptrs.push_back(jacobian_source_pose.data());
        jac_ptrs.push_back(jacobian_target_pose.data());
        jac_ptrs.push_back(jacobian_intrinsics.data());
        Eigen::Matrix<double, 16, 3> jacobian_analytic_plane;
        Eigen::Matrix<double, Sophus::SE3d::num_parameters, 16> jacobian_analytic_source_pose;
        Eigen::Matrix<double, Sophus::SE3d::num_parameters, 16> jacobian_analytic_target_pose;
        Eigen::Matrix<double, 16, RadialCamera<double>::N> jacobian_analytic_intrinsics;
        std::vector<double*> jac_ptrs_analytic;
        jac_ptrs_analytic.push_back(jacobian_analytic_plane.data());
        jac_ptrs_analytic.push_back(jacobian_analytic_source_pose.data());
        jac_ptrs_analytic.push_back(jacobian_analytic_target_pose.data());
        jac_ptrs_analytic.push_back(jacobian_analytic_intrinsics.data());
        
        Sophus::SO3d so;
        // TEST rotation jacobian individually
        Eigen::Matrix<double, Sophus::SE3d::num_parameters, 3> jacobian_pose;
        Sophus::SE3d pose = data.poses[1];
        //pose.setQuaternion({1., 0., 0., 1.});
        double* jac_ptr = jacobian_pose.data();
        double* pose_ptr = pose.data();
        Eigen::Vector3d vec{1., 1., 1.};
        ceres::CostFunction* rotation_cost = new RotationCost(vec);
        ceres::CostFunction* rotation_cost_num = 
                new ceres::AutoDiffCostFunction<RotationFunctor,
                                                3, Sophus::SE3d::num_parameters> (new RotationFunctor(vec));
        rotation_cost->Evaluate(&pose_ptr, residuals.data(), &jac_ptr);
        //std::cout << jacobian_pose.transpose() << std::endl;
        //delete *rotation_cost;
        
        rotation_cost_num->Evaluate(&pose_ptr, residuals.data(), &jac_ptr);
        //std::cout << jacobian_pose.transpose() << std::endl;
        delete rotation_cost;
        delete rotation_cost_num;
        //---------------------------------------------------------------
        {
            ceres::CostFunction* correspondence_cost = new CorrespondenceCost(source_pixels[0],
                                                                          data.planes[0],
                                                                          data.poses[data.source_views[0]],
                                                                          data.poses[data.visibilities[0][1]]);
            ceres::NumericDiffOptions num_options;
            num_options.relative_step_size = 1e-6;
            
            ceres::CostFunction* correspondence_cost_num = 
                    new ceres::NumericDiffCostFunction<CorrespondenceFunctor, ceres::NumericDiffMethodType::CENTRAL, 
                                                       32, RadialCamera<double>::N> (
                        new CorrespondenceFunctor(source_pixels[0],
                                                  data.planes[0],
                                                  data.poses[data.source_views[0]],
                                                  data.poses[data.visibilities[0][1]]),
                        ceres::TAKE_OWNERSHIP, 32, num_options);
            
            Eigen::Matrix<double, RadialCamera<double>::N, 32> jacobian;
            Eigen::Matrix<double, 2, 16> res;
            double* jac_ptr = jacobian.data();
            double* intrinsics_ptr = data.camera_param.data();
            std::cout << source_pixels[0] << std::endl;
            std::cout << "correspondence" << std::endl;
            correspondence_cost->Evaluate(&intrinsics_ptr, res.data(), &jac_ptr);
            std::cout << jacobian.block<RadialCamera<double>::N, 8>(0, 0) << std::endl;
            std::cout << res << std::endl;
            std::cout << "=========================" << std::endl;
            Eigen::Matrix<double, RadialCamera<double>::N, 32> jacobian_cp = jacobian;
            correspondence_cost_num->Evaluate(&intrinsics_ptr, res.data(), &jac_ptr);
            std::cout << jacobian.block<RadialCamera<double>::N, 8>(0, 0) << std::endl;
            std::cout << res << std::endl;
            std::cout << "=========================" << std::endl;
            std::cout << (jacobian - jacobian_cp).block<RadialCamera<double>::N, 8>(0, 0) << std::endl;
        }
        {
        Eigen::Matrix<double, 6, 3> jacobian_intrinsics;
        double* jac_intrinsics_ptr = jacobian_intrinsics.data();
        RadialCamera<double>::VecN cam_param{data.camera_param};
        cam_param[5] = -0.01;
        double* intrinsics_ptr = cam_param.data();
        Eigen::Vector3d projection_residual;
        Eigen::Vector2d X{400., 500.};
        ceres::CostFunction* projection_cost = new UnprojectionCost(X);
        ceres::CostFunction* projection_cost_num = 
                new ceres::AutoDiffCostFunction<UnprojectionFunctor, 3, RadialCamera<double>::N> (
                    new UnprojectionFunctor(X));
        std::cout << "unprojection" << std::endl;
        projection_cost->Evaluate(&intrinsics_ptr, projection_residual.data(), &jac_intrinsics_ptr);
        std::cout << jacobian_intrinsics << std::endl;
        
        projection_cost_num->Evaluate(&intrinsics_ptr, projection_residual.data(), &jac_intrinsics_ptr);
        std::cout << jacobian_intrinsics << std::endl;
        
        delete projection_cost;
        delete projection_cost_num;
        RadialCamera<double> cam{cam_param};
        
        std::cout << X.transpose() << std::endl;
        std::cout << cam.unproject(X).transpose() << std::endl;
        std::cout << cam.project(cam.unproject(X)).transpose() << std::endl;
        
        std::cout << "-------------" << std::endl;
        }
        std::cout << "more jacobians" << std::endl;
        problem_automatic.EvaluateResidualBlock(ids_automatic[i], 
                                                false, 
                                                &cost,
                                                residuals.data(),
                                                jac_ptrs.data());
        
        //std::cout << jacobian_source_pose << std::endl;
        //std::cout << residuals.transpose() << std::endl;
        std::cout << "-------------------" << std::endl;
        
        problem_numeric.EvaluateResidualBlock(ids_numeric[i], 
                                                false, 
                                                &cost,
                                                residuals.data(),
                                                jac_ptrs.data());
        //std::cout << jacobian_plane << std::endl;
        //std::cout << jacobian_source_pose << std::endl;
        //std::cout << jacobian_target_pose << std::endl;
        //std::cout << residuals.transpose() << std::endl;
        std::cout << "-------------------" << std::endl;
        
        problem_analytic.EvaluateResidualBlock(ids_analytic[i], 
                                                false, 
                                                &cost,
                                                residuals.data(),
                                                &jac_ptrs_analytic[0]);
        //std::cout << jacobian_analytic_plane << std::endl;
        //std::cout << jacobian_analytic_source_pose << std::endl;
        //std::cout << jacobian_analytic_target_pose << std::endl;
        std::cout << "---" << std::endl;
        std::cout << jacobian_intrinsics << std::endl;
        std::cout << "---" << std::endl;
        std::cout << jacobian_analytic_intrinsics<< std::endl;
        std::cout << "---" << std::endl;
        std::cout << jacobian_analytic_intrinsics - jacobian_intrinsics << std::endl;
        //std::cout << jacobian_analytic_plane - jacobian_plane << std::endl;
        std::cout << "-------------------" << std::endl;
        
    }
    // TODO: test ncc alone
    
    for (int i = 1; i < 0; i++) {
        // TODO: calculate actual patch
        Eigen::Matrix<double, 2, 16> target_pixels;
        RadialCamera<double> cam{data.camera_param};
        correspondence(source_pixels[0], data.planes[0],
                       data.poses[data.source_views[0]], data.poses[data.visibilities[0][i]], cam,
                       target_pixels);
        Eigen::Matrix<double, 16, 1> target_patch;
        for (int j = 0; j < 16; j++) {
            data.image_pyramids[data.visibilities[0][i]][level].Evaluate(target_pixels.col(j)[1],
                                                                         target_pixels.col(j)[0],
                                                                         &target_patch[j]);
        }
        
        double cost;
        Eigen::Matrix<double, 16, 1> residuals;
        Eigen::Matrix<double, 16, 16> jacobian_1;
        Eigen::Matrix<double, 16, 16> jacobian_2;
        double* jac_1_ptr = jacobian_1.data();
        double* jac_2_ptr = jacobian_2.data();
        double* param_ptr = target_patch.data();
        std::cout << "to normalize: " << std::endl;
        std::cout << target_patch << std::endl;
        ceres::CostFunction* cost_numeric = new ceres::NumericDiffCostFunction<
                                                NCCFunctor, 
                                                ceres::NumericDiffMethodType::CENTRAL, 
                                                16, 16>(new NCCFunctor);
        cost_numeric->Evaluate(&param_ptr, residuals.data(), &jac_1_ptr);
        
        std::cout << "-------" << std::endl;
        
        ceres::CostFunction* cost_analytic = new NCCCost();
        cost_analytic->Evaluate(&param_ptr, residuals.data(), &jac_2_ptr);
        std::cout << (jacobian_1 - jacobian_2).sum() << std::endl;
        std::cout << jacobian_1 << std::endl;
        std::cout << "--------" << std::endl;
        std::cout << jacobian_2 << std::endl;
    }
    
}

#endif // TEST_JACOBIANS_H
