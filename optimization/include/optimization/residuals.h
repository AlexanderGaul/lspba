#ifndef RESIDUALS_H
#define RESIDUALS_H


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <sophus/se3.hpp>
#include <basalt/image/image.h>

#include <optimization/camera.h>
#include <optimization/steps.h>
#include <optimization/problem_parameters.h>
#include <optimization/patchplt.h>

// Cost Functor for Robust Mean
struct DifferenceCostFunctor {
    DifferenceCostFunctor(double* patch) : patch(patch) {}
    template <typename T>
    bool operator() (const T* const mu, T* residual) const {
        for (int i = 0; i < 16; i++) {
            residual[i] = mu[i] - patch[i];
        }
        return true;
    }
    double* patch;
};

struct DistanceCostFunctor {
    DistanceCostFunctor(double distance) : distance(distance) {}
    template <typename T>
    bool operator() (const T* const spose_1, const T* const spose_2, T* residual) const {
        Eigen::Map<Sophus::SE3<T> const> const pose_1{spose_1};
        Eigen::Map<Sophus::SE3<T> const> const pose_2{spose_2};
        residual[0] = (pose_1.translation() - pose_2.translation()).norm() - distance;
        return true;
    }
    double distance;
};

struct AngleCostFunctor {
    AngleCostFunctor(Eigen::Vector3d v) : v(v.normalized()) {}
    
    template <typename T>
    bool operator() (const T* const su, T* residual) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> const u{su};
        residual[0] = T(10.) * (T(1.) - v.cast<T>().dot(u.normalized()));
        return true;
    }
    
    Eigen::Vector3d v;
};

// TODO: what number for template arguemtns
class ComputeInterpolationFunction : public ceres::SizedCostFunction<1, 2> {
public:
    ComputeInterpolationFunction(const basalt::Image<double>* image) : image(image) {}
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        if (!jacobians) {
            residuals[0] = image->interp(parameters[0][0], parameters[0][1]);
        } else {
            Eigen::Matrix<double, 3, 1> interp = image->interpGrad(parameters[0][0], parameters[0][1]);
            residuals[0] = interp[0];
            jacobians[0][0] = interp[1];
            jacobians[0][1] = interp[2];
        }
        return true;
    }
    
    const basalt::Image<double>* image;
};

struct PatchResidualFunctor {
    PatchResidualFunctor(const double* source_patch, const double* source_pixels, 
                  const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator,
                  const basalt::Image<double>* basalt_image) : 
        source_patch(source_patch), source_pixels(source_pixels), 
        interpolator(interpolator), basalt_image(basalt_image) {
        if (basalt_image) {
            compute_interpolation.reset(
                new ceres::CostFunctionToFunctor<1, 2>(new ComputeInterpolationFunction(basalt_image)));
        }
    }
    

    template <typename T>
    bool operator() (const T* const splane, 
                     const T* const spose_i, const T* const spose_j,
                     const T* const scamera_i, const T* scamera_j, 
                     T* sresidual) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> const plane{splane};
        Eigen::Map<Sophus::SE3<T> const> const pose_i{spose_i};
        Eigen::Map<Sophus::SE3<T> const> const pose_j{spose_j};
        Sophus::SE3<T> pose_ij{pose_j * pose_i.inverse()};
        auto pose_ij_rotation = pose_ij.rotationMatrix();
        auto pose_ij_translation = pose_ij.translation();
        
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(scamera_i);
        std::shared_ptr<RadialCamera<T>> camera(new RadialCamera<T>(intrinsics));
        
        //Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(scamera_j);
        //std::shared_ptr<RadialCamera<T>> camera_i(new RadialCamera<T>(intrinsics));
  
        
        Eigen::Matrix<T, 2, 16> x_i;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 16; j++) {
                x_i.row(i)[j] = T(source_pixels.row(i)[j]);
            }
        }
        Eigen::Matrix<T, 2, 16> x_j = correspondence(x_i, plane, pose_i, pose_j, *camera);
        /*
        for (int i = 0; i < 16; i++) {
            Eigen::Matrix<T, 2, 1> p;
            p[0] = T(source_pixels.col(i)[0]);
            p[1] = T(source_pixels.col(i)[1]);
            Eigen::Matrix<T, 3, 1> x_bar{camera->unproject(p)};
            x_j.col(i) = camera->project(pose_ij_rotation * (x_bar / (plane.transpose() * x_bar)) + pose_ij_translation);
        } */
        
        Eigen::Matrix<T, 16, 1> patch_j;
        for (int i = 0; i < 16; i++) {
            // TODO: completely discard if any location is outside the image???
            patch_j[i] = T(0.);
            if (interpolator) {
                interpolator->Evaluate((x_j.col(i)[1]), 
                                       (x_j.col(i)[0]), &patch_j[i]);
            } else {
                Eigen::Matrix<T, 2, 1> x_j_i{x_j.col(i)[0], x_j.col(i)[1]};
                if (x_j_i[0] < T(0.)) x_j_i[0] = T(0.);
                if (x_j_i[1] < T(0.)) x_j_i[1] = T(0.);
                if (x_j_i[0] >= T(basalt_image->w)) x_j_i[0] = T((double)basalt_image->w-1.);
                if (x_j_i[1] >= T(basalt_image->h)) x_j_i[1] = T((double)basalt_image->h-1.);
                (*compute_interpolation)(&x_j_i[0], &patch_j[i]);
            }
        }
        /*
        T mu = patch_j.mean();
        Eigen::Matrix<T, 16, 1> patch_j_centered = 
            patch_j - Eigen::Matrix<T, 16, 1>::Ones() * mu; */
        Eigen::Matrix<T, 16, 1> patch_j_normalized = ncc(patch_j); //patch_j_centered.normalized();

        for (int i = 0; i < 16; i++) {
            sresidual[i] = patch_j_normalized[i] - T(source_patch[i]);
        }
        return true;
    }
    
    
    // TODO: can we turn these int references???
    Eigen::Map<const Eigen::Matrix<double, 16, 1>> source_patch;
    Eigen::Map<const Eigen::Matrix<double, 2, 16>> source_pixels;
    
    const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator;
    const basalt::Image<double>* basalt_image;
    std::unique_ptr<ceres::CostFunctionToFunctor<1, 2>> compute_interpolation;
};

struct PatchResidual {
    PatchResidual(const double* source_patch, const double* source_pixels, 
                  const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator,
                  const basalt::Image<double>* basalt_image) : 
        source_patch(source_patch), source_pixels(source_pixels), 
        interpolator(interpolator), basalt_image(basalt_image) {
        if (basalt_image) {
            compute_interpolation.reset(
                new ceres::CostFunctionToFunctor<1, 2>(new ComputeInterpolationFunction(basalt_image)));
        }
    }
    

    template <typename T>
    bool operator() (const T* const splane, 
                     const T* const spose_i, const T* const spose_j,
                     const T* const scamera, 
                     T* sresidual) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> const plane{splane};
        Eigen::Map<Sophus::SE3<T> const> const pose_i{spose_i};
        Eigen::Map<Sophus::SE3<T> const> const pose_j{spose_j};
        Sophus::SE3<T> pose_ij{pose_j * pose_i.inverse()};
        auto pose_ij_rotation = pose_ij.rotationMatrix();
        auto pose_ij_translation = pose_ij.translation();
        
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(scamera);
        // TODO: do we need pointer here
        std::shared_ptr<RadialCamera<T>> camera(new RadialCamera<T>(intrinsics));
  
        
        Eigen::Matrix<T, 2, 16> x_i;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 16; j++) {
                x_i.row(i)[j] = T(source_pixels.row(i)[j]);
            }
        }
        Eigen::Matrix<T, 2, 16> x_j = correspondence(x_i, plane, pose_i, pose_j, *camera);
        /*
        for (int i = 0; i < 16; i++) {
            Eigen::Matrix<T, 2, 1> p;
            p[0] = T(source_pixels.col(i)[0]);
            p[1] = T(source_pixels.col(i)[1]);
            Eigen::Matrix<T, 3, 1> x_bar{camera->unproject(p)};
            x_j.col(i) = camera->project(pose_ij_rotation * (x_bar / (plane.transpose() * x_bar)) + pose_ij_translation);
        } */
        
        Eigen::Matrix<T, 16, 1> patch_j;
        for (int i = 0; i < 16; i++) {
            // TODO: completely discard if any location is outside the image???
            patch_j[i] = T(0.);
            if (interpolator) {
                interpolator->Evaluate((x_j.col(i)[1]), 
                                       (x_j.col(i)[0]), &patch_j[i]);
            } else {
                Eigen::Matrix<T, 2, 1> x_j_i{x_j.col(i)[0], x_j.col(i)[1]};
                if (x_j_i[0] < T(0.)) x_j_i[0] = T(0.);
                if (x_j_i[1] < T(0.)) x_j_i[1] = T(0.);
                if (x_j_i[0] >= T(basalt_image->w)) x_j_i[0] = T((double)basalt_image->w-1.);
                if (x_j_i[1] >= T(basalt_image->h)) x_j_i[1] = T((double)basalt_image->h-1.);
                (*compute_interpolation)(&x_j_i[0], &patch_j[i]);
            }
        }
        /*
        T mu = patch_j.mean();
        Eigen::Matrix<T, 16, 1> patch_j_centered = 
            patch_j - Eigen::Matrix<T, 16, 1>::Ones() * mu; */
        Eigen::Matrix<T, 16, 1> patch_j_normalized = ncc(patch_j); //patch_j_centered.normalized();

        for (int i = 0; i < 16; i++) {
            sresidual[i] = patch_j_normalized[i] - T(source_patch[i]);
        }
        return true;
    }
    
    
    // TODO: can we turn these int references???
    Eigen::Map<const Eigen::Matrix<double, 16, 1>> source_patch;
    Eigen::Map<const Eigen::Matrix<double, 2, 16>> source_pixels;
    
    const ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator;
    const basalt::Image<double>* basalt_image;
    std::unique_ptr<ceres::CostFunctionToFunctor<1, 2>> compute_interpolation;
};
/*
class PatchResidualCost : 
        public ceres::SizedCostFunction<16, 
                                        3, 
                                        Sophus::SE3d::num_parameters,
                                        Sophus::SE3d::num_parameters,
                                        RadialCamera<double>::N,
                                        RadialCamera<double>::N> {
public:
    PatchResidualCost(
            const Eigen::Matrix<double, 2, 16>& source_pixels,
            const Eigen::Matrix<double, 16, 1>& source_patch,
            ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator,
            basalt::Image<double>* basalt_image, 
            long landmark_index=-1, int visibility_index=-1,
            ProblemParameters* parameters=nullptr) : 
        pixels_source(source_pixels), patch_source(source_patch),
        interpolator(interpolator), basalt_image(basalt_image),
        landmark_index(landmark_index), visibility_index(visibility_index), parameters(parameters) {
        
        // mutable_parameter_block_sizes()->push_back(3);
        // set_num_residuals(16);
    }
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<Eigen::Vector3d const> plane{parameters[0]};
        
        Eigen::Map<Sophus::SE3d const> pose_source{parameters[1]};
        Eigen::Map<Sophus::SE3d const> pose_target{parameters[2]};
        
        Eigen::Map<RadialCamera<double>::VecN const> intrinsics_source{parameters[3]};
        std::shared_ptr<RadialCamera<double>> cam_source{new RadialCamera<double>(intrinsics_source)};
        Eigen::Map<RadialCamera<double>::VecN const> intrinsics_target{parameters[4]};
        std::shared_ptr<RadialCamera<double>> cam_target{new RadialCamera<double>(intrinsics_target)};
        
        std::vector<Eigen::Matrix<double, 2, 3>> dpixel_dplane(16);
        std::vector<Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>> dpixel_dpose_source(16);
        std::vector<Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>> dpixel_dpose_target(16);
        std::vector<Eigen::Matrix<double, 2, RadialCamera<double>::N>> dpixel_dintrinsics_source(16);
        std::vector<Eigen::Matrix<double, 2, RadialCamera<double>::N>> dpixel_dintrinsics_target(16);
        
        Eigen::Matrix<double, 2, 16> pixels_target = correspondence(pixels_source, plane, 
                       pose_source, pose_target, 
                       *camera_source, *camera_target,
                       &dpixel_dplane[0], 
                       &dpixel_dpose_source[0], 
                       &dpixel_dpose_target[0],
                       &dpixel_dintrinsics_source[0],
                       &dpixel_dintrinsics_target[0]); //jacobians ?  : nullptr);
        
        Eigen::Matrix<double, 16, 1> target_patch;
        Eigen::Matrix<double, 16, 2> jac_patch_pixel;
        
        if (interpolator) {
            for (int i = 0; i < 16; i++) {
                interpolator->Evaluate(target_pixels.col(i)[1],
                                       target_pixels.col(i)[0],
                                       &target_patch[i],
                                       jacobians ? &jac_patch_pixel.row(i)[1] : nullptr,
                                       jacobians ? &jac_patch_pixel.row(i)[0] : nullptr);
            }
        } else {
            for (int i = 0; i < 16; i++) {
                Eigen::Vector2d p{target_pixels.col(i)[0],
                                  target_pixels.col(i)[1]};
                bool jacobian_out_of_range = false;
                if (p[0] < 0) p[0] = 0.;
                if (p[1] < 0) p[1] = 0.;
                // Basalt image does not recognize exact integer coordinates 
                if (p[0] >= basalt_image->w - 1) 
                    p[0] = (double)basalt_image->w - 1.0001;
                if (p[1] >= basalt_image->h - 1) 
                    p[1] = (double)basalt_image->h - 1.0001;
                Eigen::Vector3d interp{0., 0., 0.};
                if (!jacobian_out_of_range)
                    interp = basalt_image->interpGrad<double>(p[0], 
                                                              p[1]);
                else
                    interp[0] = basalt_image->interp(p);
                target_patch[i] = interp[0];
                
                if (jacobians) {
                    jac_patch_pixel.row(i)[0] = interp[1];
                    jac_patch_pixel.row(i)[1] = interp[2];
                }
            }
        }
        Eigen::Matrix<double, 16, 1> target_patch_ncc;
        Eigen::Matrix<double, 16, 16> jac_ncc_patch;
        ncc(target_patch, target_patch_ncc, &jac_ncc_patch);
        
        for (int i = 0; i < 16; i++) {
            residuals[i] = target_patch_ncc[i] - source_patch[i];
        }
        if (jacobians) {
            Eigen::Matrix<double, 16, 3> jac_patch_plane;
            for (int i = 0; i < 16; i++) {
                jac_patch_plane.row(i) = jac_patch_pixel.row(i) * jac_pixel_plane[i];
            }
            Eigen::Matrix<double, 16, 3> jac_ncc_plane = jac_ncc_patch * jac_patch_plane;
                
            if (jacobians[0]) {
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 3; j++) {
                        jacobians[0][i * 3 + j] = jac_ncc_plane.row(i)[j];
                    }
                }
            }
            if (jacobians[1]) {
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_patch_source_pose;
                for (int i = 0; i < 16; i++) {
                    jac_patch_source_pose.row(i) = jac_patch_pixel.row(i) * jac_pixel_source_pose[i];
                }
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_ncc_source_pose = 
                    jac_ncc_patch * jac_patch_source_pose;
                
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < Sophus::SE3d::num_parameters; j++) {
                        jacobians[1][i * Sophus::SE3d::num_parameters + j] = jac_ncc_source_pose.row(i)[j];
                    }
                }
            }
            if (jacobians[2]) {
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_patch_target_pose;
                for (int i = 0; i < 16; i++) {
                    jac_patch_target_pose.row(i) = jac_patch_pixel.row(i) * jac_pixel_target_pose[i];
                }
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters, Eigen::RowMajor> jac_ncc_target_pose = 
                    jac_ncc_patch * jac_patch_target_pose;
                
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < Sophus::SE3d::num_parameters; j++) {
                        jacobians[2][i * Sophus::SE3d::num_parameters + j] = jac_ncc_target_pose.row(i)[j];
                    }
                }
            }
            if (jacobians[3]) {
                Eigen::Matrix<double, 16, RadialCamera<double>::N> jac_patch_intrinsics;
                for (int i = 0; i < 16; i++) {
                    jac_patch_intrinsics.row(i) = jac_patch_pixel.row(i) * jac_pixel_intrinsics[i];
                }
                Eigen::Matrix<double, 16, RadialCamera<double>::N> jac_ncc_intrinsics = 
                    jac_ncc_patch * jac_patch_intrinsics;
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < RadialCamera<double>::N; j++) {
                        jacobians[3][i * RadialCamera<double>::N + j] = jac_ncc_intrinsics.row(i)[j];//jac_ncc_intrinsics.row(i)[j];
                    }
                }
            }
        }
        
        return true;
    }
    
    
protected:
    const Eigen::Matrix<double, 16, 1>& patch_source;
    const Eigen::Matrix<double, 2, 16>& pixels_source;
    
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator;
    basalt::Image<double>* basalt_image;
    
    long landmark_index;
    int visibility_index;
    problem_parameters* parameters;
};
*/
class PatchResidualStructureAnalytic : 
        public ceres::SizedCostFunction<16, 
                                        3, 
                                        Sophus::SE3d::num_parameters,
                                        Sophus::SE3d::num_parameters,
                                        RadialCamera<double>::N> {
public:
    PatchResidualStructureAnalytic(
            const Eigen::Matrix<double, 2, 16>& source_pixels,
            const Eigen::Matrix<double, 16, 1>& source_patch,
            ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator,
            basalt::Image<double>* basalt_image, 
            long landmark_index=-1, int visibility_index=-1,
            ProblemParameters* parameters=nullptr) : 
        source_pixels(source_pixels), source_patch(source_patch),
        interpolator(interpolator), basalt_image(basalt_image),
        landmark_index(landmark_index), visibility_index(visibility_index), parameters(parameters) {
        
        // mutable_parameter_block_sizes()->push_back(3);
        // set_num_residuals(16);
    }
    
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        Eigen::Map<Eigen::Vector3d const> plane{parameters[0]};
        Eigen::Map<Sophus::SE3d const> source_pose{parameters[1]};
        Eigen::Map<Sophus::SE3d const> target_pose{parameters[2]};
        Eigen::Map<RadialCamera<double>::VecN const> camera_intrinsics{parameters[3]};
        std::shared_ptr<RadialCamera<double>> camera{new RadialCamera<double>(camera_intrinsics)};
        Eigen::Matrix<double, 2, 16> target_pixels;
        
        std::vector<Eigen::Matrix<double, 2, 3>> jac_pixel_plane(16);
        std::vector<Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>> jac_pixel_source_pose(16);
        std::vector<Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>> jac_pixel_target_pose(16);
        std::vector<Eigen::Matrix<double, 2, RadialCamera<double>::N>> jac_pixel_intrinsics(16);
        
        correspondence(source_pixels, plane, 
                       source_pose, target_pose, *camera, 
                       target_pixels, 
                       &jac_pixel_plane[0], 
                       &jac_pixel_source_pose[0], 
                       &jac_pixel_target_pose[0],
                       &jac_pixel_intrinsics[0]); //jacobians ?  : nullptr);
        
        Eigen::Matrix<double, 16, 1> target_patch;
        Eigen::Matrix<double, 16, 2> jac_patch_pixel;
        
        if (false) {
            for (int i = 0; i < 16; i++) {
                interpolator->Evaluate(target_pixels.col(i)[1],
                                       target_pixels.col(i)[0],
                                       &target_patch[i],
                                       jacobians ? &jac_patch_pixel.row(i)[1] : nullptr,
                                       jacobians ? &jac_patch_pixel.row(i)[0] : nullptr);
            }
        } else {
            for (int i = 0; i < 16; i++) {
                Eigen::Vector2d p{target_pixels.col(i)[0],
                                  target_pixels.col(i)[1]};
                bool jacobian_out_of_range = false;
                if (p[0] < 0) p[0] = 0.;
                if (p[1] < 0) p[1] = 0.;
                // Basalt image does not recognize exact integer coordinates 
                if (p[0] >= basalt_image->w - 1) 
                    p[0] = (double)basalt_image->w - 1.0001;
                if (p[1] >= basalt_image->h - 1) 
                    p[1] = (double)basalt_image->h - 1.0001;
                Eigen::Vector3d interp{0., 0., 0.};
                if (!jacobian_out_of_range)
                    interp = basalt_image->interpGrad<double>(p[0], 
                                                              p[1]);
                else
                    interp[0] = basalt_image->interp(p);
                target_patch[i] = interp[0];
                
                if (jacobians) {
                    jac_patch_pixel.row(i)[0] = interp[1];
                    jac_patch_pixel.row(i)[1] = interp[2];
                }
            }
        }
        Eigen::Matrix<double, 16, 1> target_patch_ncc;
        Eigen::Matrix<double, 16, 16> jac_ncc_patch;
        ncc(target_patch, target_patch_ncc, &jac_ncc_patch);
        
        for (int i = 0; i < 16; i++) {
            residuals[i] = target_patch_ncc[i] - source_patch[i];
        }
        if (jacobians) {
            Eigen::Matrix<double, 16, 3> jac_patch_plane;
            for (int i = 0; i < 16; i++) {
                jac_patch_plane.row(i) = jac_patch_pixel.row(i) * jac_pixel_plane[i];
            }
            Eigen::Matrix<double, 16, 3> jac_ncc_plane = jac_ncc_patch * jac_patch_plane;
                
            if (jacobians[0]) {
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 3; j++) {
                        jacobians[0][i * 3 + j] = jac_ncc_plane.row(i)[j];
                    }
                }
            }
            if (jacobians[1]) {
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_patch_source_pose;
                for (int i = 0; i < 16; i++) {
                    jac_patch_source_pose.row(i) = jac_patch_pixel.row(i) * jac_pixel_source_pose[i];
                }
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_ncc_source_pose = 
                    jac_ncc_patch * jac_patch_source_pose;
                
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < Sophus::SE3d::num_parameters; j++) {
                        jacobians[1][i * Sophus::SE3d::num_parameters + j] = jac_ncc_source_pose.row(i)[j];
                    }
                }
            }
            if (jacobians[2]) {
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters> jac_patch_target_pose;
                for (int i = 0; i < 16; i++) {
                    jac_patch_target_pose.row(i) = jac_patch_pixel.row(i) * jac_pixel_target_pose[i];
                }
                Eigen::Matrix<double, 16, Sophus::SE3d::num_parameters, Eigen::RowMajor> jac_ncc_target_pose = 
                    jac_ncc_patch * jac_patch_target_pose;
                
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < Sophus::SE3d::num_parameters; j++) {
                        jacobians[2][i * Sophus::SE3d::num_parameters + j] = jac_ncc_target_pose.row(i)[j];
                    }
                }
            }
            if (jacobians[3]) {
                Eigen::Matrix<double, 16, RadialCamera<double>::N> jac_patch_intrinsics;
                for (int i = 0; i < 16; i++) {
                    jac_patch_intrinsics.row(i) = jac_patch_pixel.row(i) * jac_pixel_intrinsics[i];
                }
                Eigen::Matrix<double, 16, RadialCamera<double>::N> jac_ncc_intrinsics = 
                    jac_ncc_patch * jac_patch_intrinsics;
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < RadialCamera<double>::N; j++) {
                        jacobians[3][i * RadialCamera<double>::N + j] = jac_ncc_intrinsics.row(i)[j];//jac_ncc_intrinsics.row(i)[j];
                    }
                }
            }
        }
        
        return true;
    }
    
    
protected:
    const Eigen::Matrix<double, 16, 1>& source_patch;
    const Eigen::Matrix<double, 2, 16>& source_pixels;
    
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator;
    basalt::Image<double>* basalt_image;
    
    long landmark_index;
    int visibility_index;
    ProblemParameters* parameters;
};



// Robust Loss
class Rho : public ceres::LossFunction {
public:
    Rho(double tau=0.5) : tau_(tau) {}
    void Evaluate(double s, double out[3]) const {
        
        out[0] = s / (s + tau_ * tau_);
        out[1] = tau_ * tau_ / ((s + tau_ * tau_) * (s + tau_ * tau_));
        out[2] = -2 * tau_ * tau_ / ((s + tau_ * tau_) * (s + tau_ * tau_) * (s + tau_ * tau_));
    }
private:
    double tau_;
};


#endif // RESIDUALS_H
