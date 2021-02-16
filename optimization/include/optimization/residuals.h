#ifndef RESIDUALS_H
#define RESIDUALS_H


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <sophus/se3.hpp>

#include <optimization/camera.h>

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

struct PatchResidual {
    /* should this type be T too? */
    /* TODO: how to supply eigen matrices??*/
    PatchResidual(double* source_patch, double* source_pixels, 
                  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator) : 
        source_patch(source_patch), source_pixels(source_pixels), interpolator(interpolator) {}
    
    /*  */
    /* pass 6 vector for infinitesimal change of rotation that is initialized with zero?? */
    /* have curreent pose in struct */
    /* How to get deltas for parameters */ 
    /* does copying keep automatic differentiation */
    template <typename T>
    bool operator() (const T* const splane, 
                     const T* const spose_i, const T* const spose_j,
                     const T* const scamera_i, const T* const scamera_j,
                     const T* sresidual) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> const plane(splane);
        Eigen::Map<Sophus::SE3<T> const> const pose_i(spose_i);
        Eigen::Map<Sophus::SE3<T> const> const pose_j(spose_j);
        
        /* TODO how to consts work here? */
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> intrinsics_i(scamera_i);
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> intrinsics_j(scamera_j);
        
        /* TODO how to use map here?? */
        std::shared_ptr<RadialCamera<T>> camera_i(new RadialCamera<T>(intrinsics_i));
        std::shared_ptr<RadialCamera<T>> camera_j(new RadialCamera<T>(intrinsics_j));
        
        Eigen::Matrix<T, 3, 16> X;
        for (int i = 0; i < 16; i++) {
            Eigen::Matrix<T, 3, 1> x_bar{camera_i.undistort(source_pixels.col(i))};
            X.col(i) = pose_i.inverse() * (x_bar / (plane.transpose() * x_bar));
        }
        Eigen::Matrix<T, 3, 16> X_C{pose_j * X};
        Eigen::Matrix<T, 2, 16> x_j;
        for (int i = 0; i < 16; i++) {
            x_j.col(i) = camera_j.project(X_C.col(i));
        }
        Eigen::Matrix<T, 16, 1> patch_j;
        for (int i = 0; i < 16; i++) {
            interpolator->Evaluate(x_j.col(i)[0], x_j.col[i][1], &patch_j[i]);
        }
        Eigen::Matrix<T, 16, 1> patch_j_normalized{(patch_j - patch_j.sum()) / (patch_j - patch_j.sum()).norm()};
        
        for (int i = 0; i < 16; i++) {
            sresidual[i] = patch_j_normalized[i] - source_patch[i];
        }
    }
    
    /* what members do we need??s */
    /* normalized source patches to not recompute */
    /* planes can change */
    /* plane parametrization */
    /* landmark is anchored to pixel in input image */
    
    /* TODO: image in */
    
    /* TODO: should this be pointers */
    /* should be normalized */
    Eigen::Map<Eigen::Matrix<double, 16, 1>> source_patch;
    Eigen::Map<Eigen::Matrix<double, 2, 16>> source_pixels;
    /* TODO: should this be pointers */
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator;
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
