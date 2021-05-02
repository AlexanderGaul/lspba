#ifndef STEPS_H
#define STEPS_H

#include <iostream>

#include <eigen3/Eigen/Dense>

#include <sophus/se3.hpp>

#include <optimization/camera.h>

// TODO: const reference
// TODO: do with template??
// TODO: pointers, references or what
template<typename T>
Eigen::Matrix<T, 16, 1> ncc(Eigen::Matrix<T, 16, 1> const & patch,  
                            Eigen::Matrix<T, 16, 16>* jacobian=nullptr) {
    Eigen::Matrix<T, 16, 1> patch_ncc;
    T mu = patch.mean();
    Eigen::Matrix<T, 16, 1> patch_centered = 
        patch - Eigen::Matrix<T, 16, 1>::Ones() * mu;
    patch_ncc = patch_centered.normalized();
    
    if (jacobian) {
        T sigma = patch_centered.norm();
        if (sigma == T(0.)){
            sigma = T(1.);
        }
        Eigen::Matrix<T, 16, 16> I = Eigen::Matrix<T, 16, 16>::Identity();
        Eigen::Matrix<T, 16, 16> Ones = Eigen::Matrix<T, 16, 16>::Ones();
        Eigen::Matrix<T, 16, 16> outer = patch_centered * patch_centered.transpose();
        *jacobian = ((I - Ones / T(16.)) - (outer - outer * (Ones / T(16.))) / (sigma*sigma)) / sigma;
    }
    
    return patch_ncc;
}


void ncc(const Eigen::Matrix<double, 16, 1>& patch, 
         Eigen::Matrix<double, 16, 1>& patch_ncc, 
         Eigen::Matrix<double, 16, 16>* jacobian=nullptr) {
    double mu = patch.mean();
    Eigen::Matrix<double, 16, 1> patch_centered = 
        patch - Eigen::Matrix<double, 16, 1>::Ones() * mu;
    patch_ncc = patch_centered.normalized();
    
    if (jacobian) {
        double sigma = patch_centered.norm();
        if (sigma == 0){
            sigma = 1;
        }
        Eigen::Matrix<double, 16, 16> I = Eigen::Matrix<double, 16, 16>::Identity();
        Eigen::Matrix<double, 16, 16> Ones = Eigen::Matrix<double, 16, 16>::Ones();
        Eigen::Matrix<double, 16, 16> outer = patch_centered * patch_centered.transpose();
        *jacobian = ((I - Ones / 16.) - (outer - outer * (Ones / 16.)) / (sigma*sigma)) / sigma;
    }
}

Eigen::Matrix<double, 4, 4> quaternion_normalize_derivative(Eigen::Quaternion<double> const & q) {
    Eigen::Vector4d q_vec{q.coeffs()};
    double norm = q.norm();
    Eigen::Matrix<double, 4, 4> jacobian;
    jacobian = Eigen::Matrix<double, 4, 4>::Identity() / norm - 
               (q_vec * q_vec.transpose()) / (norm * norm * norm);
    return jacobian;
}

template<typename T>
Eigen::Matrix<T, 3, 3> skew(Eigen::Matrix<T, 3, 1> const & v) {
    Eigen::Matrix<T, 3, 3> v_skew;
    v_skew << T(0.), -v[2], v[1],
              v[2], T(0.), -v[0],
              -v[1], v[0], T(0.);
    return v_skew;
}

template<typename T>
Eigen::Matrix<T, 3, 4> quaternion_vector_derivative(Eigen::Quaternion<T> const & q,
                                                    Eigen::Matrix<T, 3, 1> const & a) {
    Eigen::Matrix<T, 4, 1> const q_vec{q.coeffs()};
    Eigen::Matrix<T, 3, 3> a_skew = skew(a);
    Eigen::Matrix<T, 3, 4> jacobian;
    Eigen::Quaternion<T> inv = q.inverse();
    Eigen::Matrix<T, 3, 1> a_rev{a[2], a[1], a[0]};

    jacobian.col(3) = T(2.) * q_vec.template head<3>().cross(a);

    jacobian.template block<3, 3>(0, 0) = - T(2.) * q_vec[3] * a_skew + 
                                 T(2.) * Eigen::Matrix<T, 3, 3>::Identity() * (q_vec.template head<3>().transpose() * a) + 
                                 T(2.) * q_vec.template head<3>() * a.transpose() - T(4.) * a * q_vec.template head<3>().transpose();
    return jacobian;
}
template<typename T>
Eigen::Matrix<T, 4, 4> quaternion_inverse_derivative(Eigen::Quaternion<T> q) {
    Eigen::Matrix<T, 4, 4> jacobian_conjugate = -Eigen::Matrix<T, 4, 4>::Identity();
    jacobian_conjugate(3, 3) = T(1.);
    T norm = q.norm();
    
    Eigen::Matrix<T, 4, 4> jacobian;
    jacobian = jacobian_conjugate / (norm * norm) - 
               T(2.) * q.conjugate().coeffs() * q.coeffs().transpose() / (norm * norm * norm);
    return jacobian;
}



template<typename T>
Eigen::Matrix<T, 2, 16> correspondence(Eigen::Matrix<T, 2, 16> const & x_source,
                                       Eigen::Map<Eigen::Matrix<T, 3, 1> const> const & plane,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_source,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_target,
                                       RadialCamera<T> const & camera,
                                       Eigen::Matrix<T, 2, 3>* jac_plane=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_source=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_target=nullptr,
                                       Eigen::Matrix<T, 2, RadialCamera<T>::N>* jac_intrinsics=nullptr) {
    Sophus::SE3<T> pose_source2target = pose_target * pose_source.inverse();
    Eigen::Matrix<T, 2, 16> x_target;
    
    for (int i = 0; i < 16; i++) {
        Eigen::Matrix<T, 2, 3> projection_jacobian;
        Eigen::Matrix<T, 3, RadialCamera<T>::N> intrinsics_unprojection_jacobian;
        Eigen::Matrix<T, 2, RadialCamera<T>::N> intrinsics_projection_jacobian;
        
        Eigen::Matrix<T, 3, 1> x_bar = camera.unproject(x_source.col(i), 
                                                 jac_intrinsics ? &intrinsics_unprojection_jacobian : nullptr);
        T inv_depth = plane.transpose() * x_bar;
        Eigen::Matrix<T, 3, 1> X_source = x_bar / inv_depth;
        Eigen::Matrix<T, 3, 1> X_target = pose_source2target * X_source;
        x_target.col(i) = camera.project(X_target, 
                                         &projection_jacobian, 
                                         jac_intrinsics ? &intrinsics_projection_jacobian : nullptr);
        
        if (jac_plane) {
            Eigen::Matrix<T, 2, 3> jac = projection_jacobian * 
                    pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                    x_bar * x_bar.transpose() / (-inv_depth * inv_depth);
            jac_plane[i] = jac;
        }
        if (jac_pose_source) {
            jac_pose_source[i].template block<2, 4>(0, 0) =
                projection_jacobian *
                pose_target.rotationMatrix() * 
                quaternion_vector_derivative<T>(pose_source.unit_quaternion().inverse(), X_source-pose_source.translation()) *
                quaternion_inverse_derivative<T>(pose_source.unit_quaternion());
            jac_pose_source[i].template block<2, 3>(0, 4) = 
                projection_jacobian * (-pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose());
        }
        if (jac_pose_target) {
            jac_pose_target[i].template block<2, 4>(0, 0) = 
                projection_jacobian *
                quaternion_vector_derivative<T>(pose_target.unit_quaternion(),
                                             pose_source.inverse() * X_source);
            jac_pose_target[i].template block<2, 3>(0, 4) = projection_jacobian;
        }
        if (jac_intrinsics) {
            jac_intrinsics[i] = 
                projection_jacobian * pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                (Eigen::Matrix<T, 3, 3>::Identity() / inv_depth - x_bar * plane.transpose() / (inv_depth * inv_depth)) * 
                intrinsics_unprojection_jacobian;
            jac_intrinsics[i] += intrinsics_projection_jacobian;
        }
    }
    
    return x_target;
}

template<typename T>
Eigen::Matrix<T, 2, 16> correspondence(Eigen::Matrix<T, 2, 16> const & x_source,
                                       Eigen::Map<Eigen::Matrix<T, 3, 1> const> const & plane,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_source,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_target,
                                       RadialCamera<T> const & camera_source,
                                       RadialCamera<T> const & camera_target,
                                       Eigen::Matrix<T, 2, 3>* jac_plane=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_source=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_target=nullptr,
                                       Eigen::Matrix<T, 2, RadialCamera<T>::N>* jac_intrinsics=nullptr) {
    Sophus::SE3<T> pose_source2target = pose_target * pose_source.inverse();
    Eigen::Matrix<T, 2, 16> x_target;
    
    for (int i = 0; i < 16; i++) {
        Eigen::Matrix<T, 2, 3> projection_jacobian;
        Eigen::Matrix<T, 3, RadialCamera<T>::N> intrinsics_unprojection_jacobian;
        Eigen::Matrix<T, 2, RadialCamera<T>::N> intrinsics_projection_jacobian;
        
        Eigen::Matrix<T, 3, 1> x_bar = camera_source.unproject(x_source.col(i), 
                                                 jac_intrinsics ? &intrinsics_unprojection_jacobian : nullptr);
        T inv_depth = plane.transpose() * x_bar;
        Eigen::Matrix<T, 3, 1> X_source = x_bar / inv_depth;
        Eigen::Matrix<T, 3, 1> X_target = pose_source2target * X_source;
        x_target.col(i) = camera_target.project(X_target, 
                                         &projection_jacobian, 
                                         jac_intrinsics ? &intrinsics_projection_jacobian : nullptr);
        
        if (jac_plane) {
            Eigen::Matrix<T, 2, 3> jac = projection_jacobian * 
                    pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                    x_bar * x_bar.transpose() / (-inv_depth * inv_depth);
            jac_plane[i] = jac;
        }
        if (jac_pose_source) {
            jac_pose_source[i].template block<2, 4>(0, 0) =
                projection_jacobian *
                pose_target.rotationMatrix() * 
                quaternion_vector_derivative<T>(pose_source.unit_quaternion().inverse(), X_source-pose_source.translation()) *
                quaternion_inverse_derivative<T>(pose_source.unit_quaternion());
            jac_pose_source[i].template block<2, 3>(0, 4) = 
                projection_jacobian * (-pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose());
        }
        if (jac_pose_target) {
            jac_pose_target[i].template block<2, 4>(0, 0) = 
                projection_jacobian *
                quaternion_vector_derivative<T>(pose_target.unit_quaternion(),
                                             pose_source.inverse() * X_source);
            jac_pose_target[i].template block<2, 3>(0, 4) = projection_jacobian;
        }
        if (jac_intrinsics) {
            jac_intrinsics[i] = 
                projection_jacobian * pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                (Eigen::Matrix<T, 3, 3>::Identity() / inv_depth - x_bar * plane.transpose() / (inv_depth * inv_depth)) * 
                intrinsics_unprojection_jacobian;
            jac_intrinsics[i] += intrinsics_projection_jacobian;
        }
    }
    
    return x_target;
}


template<typename T>
Eigen::Matrix<T, 2, 16> correspondence(Eigen::Matrix<T, 2, 16> const & x_source,
                                       Eigen::Map<Eigen::Matrix<T, 3, 1> const> const & plane,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_source,
                                       Eigen::Map<Sophus::SE3<T> const> const & pose_target,
                                       RadialCamera<T> const & camera_source,
                                       RadialCamera<T> const & camera_target,
                                       Eigen::Matrix<T, 2, 3>* jac_plane=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_source=nullptr,
                                       Eigen::Matrix<T, 2, Sophus::SE3<T>::num_parameters>* jac_pose_target=nullptr,
                                       Eigen::Matrix<T, 2, RadialCamera<T>::N>* jac_intrinsics_source=nullptr,
                                       Eigen::Matrix<T, 2, RadialCamera<T>::N>* jac_intrinsics_target=nullptr) {
    
}



// TODO: template
// TODO: return output
void correspondence(const Eigen::Matrix<double, 2, 16>& x_source,
                    const Eigen::Vector3d& plane,
                    const Sophus::SE3d& pose_source,
                    const Sophus::SE3d& pose_target,
                    const RadialCamera<double>& camera,
                    Eigen::Matrix<double, 2, 16>& x_target,
                    Eigen::Matrix<double, 2, 3>* jac_plane=nullptr,
                    Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>* jac_pose_source=nullptr,
                    Eigen::Matrix<double, 2, Sophus::SE3d::num_parameters>* jac_pose_target=nullptr,
                    Eigen::Matrix<double, 2, RadialCamera<double>::N>* jac_intrinsics=nullptr
                    ) {
    Sophus::SE3d pose_source2target = pose_target * pose_source.inverse();
    
    for (int i = 0; i < 16; i++) {
        Eigen::Matrix<double, 2, 3> projection_jacobian;
        Eigen::Matrix<double, 3, RadialCamera<double>::N> intrinsics_unprojection_jacobian;
        Eigen::Matrix<double, 2, RadialCamera<double>::N> intrinsics_projection_jacobian;
        
        Eigen::Vector3d x_bar = camera.unproject(x_source.col(i), &intrinsics_unprojection_jacobian); // don't require derivative right now
        double inv_depth = plane.transpose() * x_bar;
        Eigen::Vector3d X_source = x_bar / inv_depth;
        Eigen::Vector3d X_target = pose_source2target * X_source;
        x_target.col(i) = camera.project(X_target, &projection_jacobian, &intrinsics_projection_jacobian);
        
        if (jac_plane) {
            Eigen::Matrix<double, 2, 3> jac = projection_jacobian * 
                    pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                    x_bar * x_bar.transpose() / (-inv_depth * inv_depth);
            jac_plane[i] = jac;
        }
        if (jac_pose_source) {
            jac_pose_source[i].block<2, 4>(0, 0) =
                projection_jacobian *
                pose_target.rotationMatrix() * 
                quaternion_vector_derivative<double>(pose_source.unit_quaternion().inverse(), X_source-pose_source.translation()) *
                quaternion_inverse_derivative(pose_source.unit_quaternion());
            jac_pose_source[i].block<2, 3>(0, 4) = 
                projection_jacobian * (-pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose());
        }
        if (jac_pose_target) {
            jac_pose_target[i].block<2, 4>(0, 0) = 
                projection_jacobian *
                quaternion_vector_derivative(pose_target.unit_quaternion(),
                                             pose_source.inverse() * X_source);
            jac_pose_target[i].block<2, 3>(0, 4) = projection_jacobian;
        }
        if (jac_intrinsics) {
            jac_intrinsics[i] = 
                projection_jacobian * pose_target.rotationMatrix() * pose_source.rotationMatrix().transpose() * 
                (Eigen::Matrix<double, 3, 3>::Identity() / inv_depth - x_bar * plane.transpose() / (inv_depth * inv_depth)) * 
                intrinsics_unprojection_jacobian;
            jac_intrinsics[i] += intrinsics_projection_jacobian;
        }
    }
        
}

#endif // STEPS_H
