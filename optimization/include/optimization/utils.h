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

/* TODO: is p in camera frame */
/* TODO: are we actually using this*/
Eigen::Vector3d get_plane(Eigen::Vector3d p, Eigen::Vector3d normal) {
    Eigen::Vector3d n{normal};
    double d = n.transpose().dot(p);
    n /= d;
    return n;
}

// TODO: do this without 3d point???
Eigen::Vector3d get_normal(Eigen::Vector3d p, Eigen::Vector3d plane, Sophus::SE3<double>& pose) {
    /* TODO: compute depth */
    Eigen::Vector3d p_C = pose * p;
    double d = p_C[2];
    Eigen::Vector3d n{plane * d};
    //n[2] = -n[2];
    return (pose.rotationMatrix().inverse() * n).normalized();
}

Eigen::Matrix<double, 3, 16> create_grid(Eigen::Vector3d point, Eigen::Vector3d normal, double scale = 0.05) {
    Eigen::Matrix<double, 3, 16> grid;
    Eigen::Vector3d horizontal{-normal[2], 0., normal[0]};
    Eigen::Vector3d vertical{normal.cross(horizontal)};
    vertical.normalize();
    horizontal.normalize();
    int i = 0;
    for (float y = -1.5; y <= 1.5; y += 1.) {
        for(float x = -1.5; x <= 1.5; x += 1.) {
            grid.col(i) = point + scale * x * horizontal + scale * y * vertical;
            i++;
        }
    }
    
    return grid;
}

void create_pyramid_images(std::vector<cv::Mat> const & images_color,
                           std::vector<std::vector<cv::Mat>> & mat_pyramids,
                           std::vector<std::vector<ceres::Grid2D<double, 1>>> * grid_pyramids = nullptr,
                           std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>>> *  interpolator_pyramids = nullptr,
                           std::vector<std::vector<basalt::Image<double>>> * basalt_pyramids = nullptr) {
    for (cv::Mat image : images_color) {
        cv::Mat im_1;
        cv::Mat im_2;
        
        cv::Mat mat_color;
        cv::cvtColor(image, mat_color, cv::ColorConversionCodes::COLOR_BGR2GRAY);
        mat_color.convertTo(im_1, CV_64F);

        cv::Mat mat_blur;
        cv::Mat mat_clone;
        
        cv::GaussianBlur(im_1, mat_blur, cv::Size(3, 3), 0);
        cv::resize(mat_blur, mat_clone, cv::Size(mat_blur.size[1]/ 2, mat_blur.size[0] / 2), 0, 0, cv::INTER_NEAREST);
        im_2 = mat_clone.clone();
        //cv::imshow("scale", mat2 / 255.);
        //cv::waitKey();
        //cv::pyrDown(mat1, mat2); //cv::BORDER_REPLICATE
        
        im_1 /= 255.;
        im_2 /= 255.;
        mat_pyramids.push_back({im_1, im_2});
        
        if (grid_pyramids) {
            grid_pyramids->push_back(
                {ceres::Grid2D<double, 1>((double*)(mat_pyramids.back()[0].data), 
                                                    0, mat_pyramids.back()[0].rows, 0, mat_pyramids.back()[0].cols),
                 ceres::Grid2D<double, 1>((double*)(mat_pyramids.back()[1].data), 
                                                    0, mat_pyramids.back()[1].rows, 0, mat_pyramids.back()[1].cols)});
            if (interpolator_pyramids) {
                interpolator_pyramids->push_back(
                    {ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids->back()[0]),
                     ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(grid_pyramids->back()[1])});
            }
        }
        if (basalt_pyramids) {
            basalt_pyramids->push_back(
                {basalt::Image<double>((double*)mat_pyramids.back()[0].data, mat_pyramids.back()[0].cols, mat_pyramids.back()[0].rows, 8*mat_pyramids.back()[0].cols),
                 basalt::Image<double>((double*)mat_pyramids.back()[1].data, mat_pyramids.back()[1].cols, mat_pyramids.back()[1].rows, 8*mat_pyramids.back()[1].cols)});
        }
    }
    
}

// TODO: move into visuializationn utils
void get_camera_symbol(Sophus::SE3d pose, RadialCamera<double> camera,
                       std::vector<Eigen::Vector3d>& points, 
                       std::vector<Eigen::Vector2i>& lines) {
    
    std::array<Eigen::Vector2d, 4> corners;
    corners[0] = Eigen::Vector2d{0, 0};
    corners[1] = Eigen::Vector2d{0, 1080};
    corners[2] = Eigen::Vector2d{1920, 1080}; 
    corners[3] = Eigen::Vector2d{1920, 0};
    std::array<Eigen::Vector3d, 4> corners_3D;
    for (int i = 0; i < 4; i++) {
        corners_3D[i] = camera.unproject(corners[i]);
        corners_3D[i] *= 0.2;
    }
    
    points.push_back(-pose.rotationMatrix().transpose() * pose.translation());
    for (int j = 0; j < 4; j++) {
        points.push_back(pose.inverse() * corners_3D[j]);
    }
    
    lines.push_back(Eigen::Vector2i{0, 1});
    lines.push_back(Eigen::Vector2i{0, 2});
    lines.push_back(Eigen::Vector2i{0, 3});
    lines.push_back(Eigen::Vector2i{0, 4});
    lines.push_back(Eigen::Vector2i{1, 2});
    lines.push_back(Eigen::Vector2i{2, 3});
    lines.push_back(Eigen::Vector2i{3, 4});
    lines.push_back(Eigen::Vector2i{4, 1});
    
}


#endif // UTILS_H
