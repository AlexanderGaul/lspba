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

struct PatchResidual {
    PatchResidual(double* source_patch, double* source_pixels, 
                  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>* interpolator) : 
        source_patch(source_patch), source_pixels(source_pixels), 
        interpolator(interpolator) {}
    
    template <typename T>
    bool operator() (const T* const spose_i, const T* const spose_j,
                     const T* const scamera, const T* const splane,
                     T* sresidual) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1> const> const plane{splane};
        Eigen::Map<Sophus::SE3<T> const> const pose_i{spose_i};
        Eigen::Map<Sophus::SE3<T> const> const pose_j{spose_j};
        Sophus::SE3<T> pose_ij{pose_j * pose_i.inverse()};
        
        Eigen::Map<Eigen::Matrix<T, 6, 1> const> const intrinsics(scamera);
        std::shared_ptr<RadialCamera<T>> camera(new RadialCamera<T>(intrinsics));
        
        Eigen::Matrix<T, 2, 16> x_j;
        for (int i = 0; i < 16; i++) {Eigen::Matrix<T, 2, 1> p;
            p[0] = T(source_pixels.col(i)[0]);
            p[1] = T(source_pixels.col(i)[1]);
            Eigen::Matrix<T, 3, 1> x_bar{camera->unproject(p)};
            x_j.col(i) = camera->project(pose_ij * (x_bar / (plane.transpose() * x_bar)));
        }
        Eigen::Matrix<T, 16, 1> patch_j;
        for (int i = 0; i < 16; i++) {
            patch_j[i] = T(0.);
            interpolator->Evaluate((x_j.col(i)[1]), 
                                   (x_j.col(i)[0]), &patch_j[i]);
        }
        T mu = T(0);
        for (int i = 0; i < 16; i++) {
            mu += patch_j[i] * T(1./16.);
        }
        Eigen::Matrix<T, 16, 1> patch_j_centered;
        for (int i = 0; i < 16; i++) {
            patch_j_centered[i] = patch_j[i] - mu;
        }
        Eigen::Matrix<T, 16, 1> patch_j_normalized = patch_j_centered.normalized();

        for (int i = 0; i < 16; i++) {
            sresidual[i] = patch_j_normalized[i] - source_patch[i];
        }
        return true;
    }
    Eigen::Map<Eigen::Matrix<double, 16, 1>> source_patch;
    Eigen::Map<Eigen::Matrix<double, 2, 16>> source_pixels;
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
