#ifndef IO_H
#define IO_H

#include <string>
#include <iostream>
#include <sstream> 
#include <filesystem>
#include <iterator>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

#include <optimization/camera.h>

bool read_camera_parameters(std::string path, RadialCamera<double>::VecN& param) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open() || file.fail()) {
        file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    /* TODO: initilaize with istream iterator */
    if (file >> param[0] &&
        file >> param[1] &&
        file >> param[2] &&
        file >> param[3] &&
        file >> param[4] &&
        file >> param[5])
        return true;
    else 
        return false;
}

/* TODO: what datatype to use exactly*/
bool read_poses(std::string path, std::vector<Sophus::SE3d>& poses) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open() || file.fail()) {
        file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        std::istringstream iss{line};
        std::vector<double> values{std::istream_iterator<double>{iss}, 
                                   std::istream_iterator<double>{}};

        /* TODO: how to do this?? */
        Eigen::Matrix3d rotation;
        rotation << values[0], values[1], values[2],
                    values[3], values[4], values[5],
                    values[6], values[7], values[8];
        Eigen::Vector3d translation;
        translation << values[9], values[10], values[11];
        Sophus::SE3d pose;
        pose.setRotationMatrix(rotation);
        pose.translation() = translation;
        poses.push_back(pose);
    }
    return true;
}

/* TODO: separate scales from landmarks */
bool read_points_normals_gridscales(std::string path,
                    std::vector<Eigen::Vector3d>& points,
                    std::vector<Eigen::Vector3d>& normals,
                    std::vector<double>& scales) {
    std::ifstream grid_file;
    grid_file.open(path, std::ios::in | std::ios::binary);
    if (!grid_file.is_open() || grid_file.fail()) {
        grid_file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    Eigen::Vector3d vec;
    double num;
    /* TODO: how to handle errors while bools and stuff*/
    /* TODO: how are objects copied? */
    while (grid_file >> vec[0]) {
        grid_file >> vec[1];
        grid_file >> vec[2];
        points.push_back(vec);
        grid_file >> vec[0];
        grid_file >> vec[1];
        grid_file >> vec[2];
        normals.push_back(vec);
        grid_file >> num;
        scales.push_back(num);
    }
    return true;
}

bool write_points_normals(std::string path,
                          std::vector<Eigen::Vector3d>& points,
                          std::vector<Eigen::Vector3d>& normals) {
    std::fstream file;
    file.open(path, std::fstream::out);
    for (int i = 0; i < std::min(points.size(), normals.size()); i++) {
        file << points[i].transpose();
        file << " ";
        file << normals[i].transpose();
        file << "\n";
    }
    file.close();
    return true;
}

bool read_visibility(std::string path,
                     std::vector<std::vector<int>>& visibility) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open() || file.fail()) {
        file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    int current_point = -1;
    int point_idx;
    int view_idx;
    while (file >> point_idx && file >> view_idx) {
        for (; current_point < point_idx; current_point++) {
            visibility.emplace_back();
        }
        visibility[current_point].push_back(view_idx);
    }
    return true;
}

bool read_landmarks(std::string path,
                    std::vector<Eigen::Vector2d>& pixels,
                    std::vector<Eigen::Vector3d>& planes) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open() || file.fail()) {
        file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    Eigen::Vector2d p;
    Eigen::Vector3d n;
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        std::istringstream iss{line};
        std::vector<double> values{std::istream_iterator<double>{iss}, 
                                   std::istream_iterator<double>{}};
        if (values.size() != 5) {
            continue; /* TODO: What to do here? */
        }
        
        p << values[0], values[1];
        n << values[2], values[3], values[4];
        
        pixels.push_back(p);
        planes.push_back(n);
    }
    return true;
}

bool read_source_frame(std::string path,
                       std::vector<int>& source_idx) {
    std::ifstream grid_file;
    grid_file.open(path, std::ios::in | std::ios::binary);
    if (!grid_file.is_open() || grid_file.fail()) {
        grid_file.close();
        std::cerr << "Could not open file stream " << path << std::endl;
        return false;
    }
    int index;
    while (grid_file >> index) {
        source_idx.push_back(index);
    }
    return true;
}

bool load_images(std::filesystem::path path, 
                 std::vector<cv::Mat>& images_color) {
    std::filesystem::directory_iterator directory_iterator(path);
    std::set<std::filesystem::path> paths;
    for (std::filesystem::directory_entry const & entry : directory_iterator) {
        paths.insert(entry.path());
    }
    int i = -1;
    for (std::filesystem::path path : paths) {
        std::string extension = path.filename().extension().string();
        if (extension != ".png" && extension != ".jpg") continue;
        int j;
        try {
            j = std::stoi(path.filename().filename().string()); 
        } catch (std::invalid_argument ex) {
            continue;
        }
        if (j > i) {
            images_color.push_back(cv::imread(path));
            i = j;
        }
    }
    return true;
}

#endif // IO_H
