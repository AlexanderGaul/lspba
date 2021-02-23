#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <optimization/camera.h>

template<typename T>
Eigen::Matrix<T, 3, 1> unproject(Eigen::Matrix<T, 2, 1>& p, 
                                 Eigen::Matrix<T, 3, 1>& plane,
                                 RadialCamera<T>& camera,
                                 Sophus::SE3<T>& pose) {
    Eigen::Matrix<T, 3, 1> x_bar{camera.unproject(p)};
    Eigen::Matrix<T, 3, 1> X{pose.inverse() * (x_bar / (plane.transpose() * x_bar))};
    return X;
}

Eigen::Vector3d get_plane(Eigen::Vector3d p, Eigen::Vector3d normal) {
    Eigen::Vector3d n{normal / normal[2]};
    double d = n.transpose().dot(p);
    n /= d;
    return n;
}


#endif // UTILS_H
