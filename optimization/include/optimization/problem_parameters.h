#ifndef PROBLEM_PARAMETERS_H
#define PROBLEM_PARAMETERS_H

#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include <sophus/se3.hpp>
#include <basalt/image/image.h>

#include <optimization/camera.h>
#include <optimization/utils.h>
#include <optimization/steps.h>


struct ProblemParameters {
    public:
    //problem_parameters() = default;
    
    ProblemParameters(std::vector<Eigen::Vector3d> planes,
                       std::vector<Sophus::SE3d> poses,
                       RadialCamera<double>::VecN camera_param,
                       std::vector<Eigen::Vector2d> pixels,
                       std::vector<int> source_views,
                       std::vector<std::vector<int>> visibilities,
                       std::vector<std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>>& image_pyramids,
                       std::vector<std::vector<basalt::Image<double>>> basalt_pyramids,
                       std::vector<cv::Mat> color_iamges={},
                       std::vector<std::vector<cv::Mat>> mat_pyramids={}) :
        planes(planes), poses(poses), camera_param(camera_param), 
        pixels(pixels), source_views(source_views), visibilities(visibilities),
        image_pyramids(image_pyramids), basalt_pyramids(basalt_pyramids),
        color_images(color_iamges), mat_pyramids(mat_pyramids)  {}
    
    // TODO: constructor based on
    ProblemParameters(ProblemParameters& param, std::vector<long> selection) :
        poses(param.poses), camera_param(param.camera_param), 
        image_pyramids(param.image_pyramids), basalt_pyramids(param.basalt_pyramids),
        color_images(param.color_images), mat_pyramids(param.mat_pyramids) {
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
    std::vector<std::vector<basalt::Image<double>>> basalt_pyramids;
    
    // TODO: turn this into pointer
    std::vector<cv::Mat> color_images;
    std::vector<std::vector<cv::Mat>> mat_pyramids;
    
    // TODO: return vector or accept vector 
    std::vector<Eigen::Vector3d> get_normals() const {
        std::vector<Eigen::Vector3d> normals;
        for (long i = 0; i < pixels.size(); i++) {
            normals.push_back(get_normal(i));
        }
        return normals;
    }
    Eigen::Vector3d get_normal(long landmark) const {
        return (poses[source_views[landmark]].rotationMatrix().inverse() * planes[landmark]).normalized();
    }
    
    void set_planes_from_normals(std::vector<Eigen::Vector3d> normals) {
        std::vector<Eigen::Vector3d> points = get_points();
        std::vector<Eigen::Vector3d> planes_new;
        for (int i = 0; i < normals.size(); i++) {
            planes_new.push_back(get_plane(poses[source_views[i]] * points[i], 
                                           poses[source_views[i]] * normals[i]));
        }
        planes = planes_new;
    }
    
    std::vector<Eigen::Vector3d> get_points() const {
        std::vector<Eigen::Vector3d> points;
        RadialCamera<double> camera{camera_param};
        for (int i = 0; i < pixels.size(); i++) {
            points.push_back(unproject(pixels[i], planes[i], camera, poses[source_views[i]]));
        }
        return points;
    }
    Eigen::Vector3d get_point(long landmark) const {
        RadialCamera<double> camera{camera_param};
        return unproject(pixels[landmark], planes[landmark], camera, poses[source_views[landmark]]);
    }
    
    std::vector<Eigen::Matrix<double, 2, 16>> get_pixel_grids(int level) const {
        std::vector<Eigen::Matrix<double, 2, 16>> pixel_grids;
        for (int i = 0; i < pixels.size(); i++) {
            // TODO: finish
        }
        return pixel_grids;
    }
    
    std::vector<Eigen::Matrix<double, 16, 1>> get_patches(int level) const { /* TODO: implement */}
    
    // TODO: figure out angle
    void filter_extreme_angles(double angle=0.) {
        std::vector<Eigen::Vector3d> normals = get_normals();
        std::vector<Eigen::Vector3d> points = get_points();
        Eigen::Vector3d z{0., 0., 1.};
        for (int i = 0; i < normals.size(); i++) { 
            for (auto view_idx_iter = visibilities[i].begin(); 
                 view_idx_iter != visibilities[i].end(); ) {
                double a = //-poses[*view_idx_iter].rotationMatrix().row(2).normalized() *
                        (poses[*view_idx_iter].rotationMatrix() * 
                         normals[i].normalized()).transpose() * z;
                double b = acos(a) / 3.14 * 180;
                double c = (3.14 / 16. * 5) / 3.14 * 180;
                if (acos(a) > 3.14 / 16. * 6) {
                    view_idx_iter = visibilities[i].erase(view_idx_iter);
                    //view_idx_iter++;
                } else {
                    view_idx_iter++;
                }
            }
        }
    }
};


// TODO: move all these into memmber functions

// TODO: pointers to allow for discarding results
void compute_patches(ProblemParameters& data,
                     int level,
                     std::vector<Eigen::Matrix<double, 2, 16>>& source_pixels,
                     std::vector<Eigen::Matrix<double, 16, 1>>& source_patches,
                     bool normalize=true
                     ) {
    double scale = pow(2, level);
    for (int l = 0; l < data.pixels.size(); l++) {
        Eigen::Matrix<double, 2, 16> grid;
        Eigen::Matrix<double, 16, 1> patch;
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                grid.col(i)[0] = data.pixels[l][0] / scale + x;
                grid.col(i)[1] = data.pixels[l][1] / scale + y;
                data.image_pyramids[data.source_views[l]][level].Evaluate((grid.col(i)[1]), 
                                                                          (grid.col(i)[0]),
                                                                          &patch[i]);
                // TODO: add basalt pyramids
                i++;
            }
        }
        source_pixels.push_back(grid);
        if (normalize) {
            Eigen::Matrix<double, 16, 1> patch_normalized;
            ncc(patch, patch_normalized, nullptr);
            source_patches.push_back(patch_normalized);
        } else {
            source_patches.push_back(patch);
        }
    }
}

void get_pixel_grids(ProblemParameters const & parameters, int level,
                     std::vector<Eigen::Matrix<double, 2, 16>> & pixel_grids) {
    double scale = pow(2, level);
    for (int l = 0; l < parameters.pixels.size(); l++) {
        Eigen::Matrix<double, 2, 16> grid;
        int i = 0;
        for (float y = -1.5; y <= 1.5; y += 1.) {
            for(float x = -1.5; x <= 1.5; x += 1.) {
                grid.col(i)[0] = parameters.pixels[l][0] / scale + x;
                grid.col(i)[1] = parameters.pixels[l][1] / scale + y;
                i++;
            }
        }
        pixel_grids.push_back(grid);
    }
}
void get_patches(ProblemParameters const & parameters, 
               std::vector<Eigen::Matrix<double, 2, 16>> const & grids,
               int level,
               std::vector<Eigen::Matrix<double, 16, 1>> patches) {
    // TODO: implement
    for (int l = 0; l < grids.size(); l++) {
        Eigen::Matrix<double, 16, 1> patch;
        for (int i = 0; i < 16; i++) {
            parameters.image_pyramids[
                parameters.source_views[l]][level].Evaluate(grids[l].col(i)[1], 
                                                            grids[l].col(i)[0],
                                                            &patch[i]);
        }
        patches.push_back(patch);
    }
}

ProblemParameters perturb_data(ProblemParameters const & parameters,
                                double min_depth=0., double max_depth=0.02,
                                double min_angle=0., double max_angle=3.14 / 12.) {
    ProblemParameters parameters_perturbed{parameters};
    RadialCamera<double> camera{parameters.camera_param};
    if (min_depth > max_depth) min_depth = max_depth;
    if (min_angle > max_angle) min_angle = max_angle;
    
    if (max_depth > 0 || min_depth > 0) {
        std::vector<Eigen::Vector3d> p3D = parameters_perturbed.get_points();
        std::vector<Eigen::Vector3d> normals_init = parameters_perturbed.get_normals();
        std::cout << normals_init[0].transpose() << std::endl;
        std::vector<Eigen::Vector3d> planes_perturbed;
        
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.01, 0.005);
        std::bernoulli_distribution plusminus(0.5);
        for (int i = 0; i < p3D.size(); i++) {
            Eigen::Vector3d p_C = 
                parameters_perturbed.poses[parameters_perturbed.source_views[i]] * p3D[i];
            double perturbation;
            if (min_depth == max_depth) {
                perturbation = max_depth;
            } else {
                perturbation = -100.;
                while (abs(perturbation) > max_depth) {
                    perturbation = distribution(generator);
                }
            }
            if (plusminus(generator)) perturbation = -perturbation;
            p_C = p_C * ((p_C[2] + perturbation) / p_C[2]);
            planes_perturbed.push_back(
                get_plane(p_C, 
                          parameters_perturbed.poses[parameters_perturbed.source_views[i]]
                              .rotationMatrix() * normals_init[i]));
        }
        parameters_perturbed.planes = planes_perturbed;
    } 

    if (max_angle > 0. || min_angle > 0.) {
        std::vector<Eigen::Vector3d> normals_init = parameters_perturbed.get_normals();
        std::vector<Eigen::Vector3d> normals_pert;
        
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(max_angle, (max_angle - min_angle) / 2.);
        std::bernoulli_distribution plusminus(0.5);
        for (int i = 0; i < normals_init.size(); i++) {
            Eigen::Vector3d normal_pert;
            Eigen::Vector3d normal = normals_init[i]; //{0., 0., 1.};
            double angle = 0.;
            do {
                double r1 = (plusminus(generator) ? 1. : -1.) * distribution(generator);
                Sophus::SE3d rot1 = Sophus::SE3d::rotX(r1);
                Sophus::SE3d rot2 = Sophus::SE3d::rotY((plusminus(generator) ? 1. : -1.) * distribution(generator));
                Sophus::SE3d rot3 = Sophus::SE3d::rotZ((plusminus(generator) ? 1. : -1.) * distribution(generator));
                Eigen::Quaternion<double> rot = (rot3 * rot1).unit_quaternion();
                normal_pert = rot * normal;
                angle = acos(normal_pert.transpose() * normal);
                if (min_angle == max_angle) {
                    Eigen::AngleAxis<double> angle_axis(rot);
                    angle_axis.angle() = angle_axis.angle() / angle * max_angle;
                    rot = Eigen::Quaternion<double>(angle_axis);
                    normal_pert = rot* normal;
                    angle = acos(normal_pert.transpose() * normal);
                    break;
                }
            }
            while (angle > max_angle || angle < min_angle);
            normals_pert.push_back(normal_pert);
        }
        parameters_perturbed.set_planes_from_normals(normals_pert);
    } 
    return parameters_perturbed;
}

void get_world_grids(ProblemParameters const & parameters,
                     double spacing,
                     std::vector<Eigen::Matrix<double, 3, 16>> & grids) {
    std::vector<Eigen::Vector3d> points = parameters.get_points();
    std::vector<Eigen::Vector3d> normals = parameters.get_normals();
    for (int l = 0; l < points.size(); l++) {
        Eigen::Matrix<double, 3, 16> grid = create_grid(points[l], normals[l], spacing);
        grids.push_back(grid);
    }
}


#endif // PROBLEM_PARAMETERS_H
