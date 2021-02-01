import numpy as np

import open3d as o3d

from scipy.spatial.transform import Rotation
from sklearn import preprocessing
import scipy.sparse
import time

def get_pose(image_data) :
    R = Rotation.from_quat([image_data.qvec[1],
                            image_data.qvec[2],
                            image_data.qvec[3],
                            image_data.qvec[0]]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = image_data.tvec
    return pose


def intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_camera_parameters(camera_data) :
    camera_parameters = {}
    camera_parameters['distortion'] = camera_data.params[3]
    # COLMAP assumes pixel centers to be at 0.5
    camera_parameters['intrinsics'] = intrinsic_matrix(camera_data.params[0],
                                                       camera_data.params[0],
                                                       camera_data.params[1]-0.5,
                                                       camera_data.params[2]-0.5)
    camera_parameters['resolution'] = np.array([camera_data.width, 
                                                camera_data.height])
    return camera_parameters


def distort_simple_radial(xys, k_1) :
    r = np.linalg.norm(xys, axis=1)
    return xys * (1 + k_1 * r * r).reshape(-1, 1)


def undistort_simple_radial(xys, k_1) :
    r = np.linalg.norm(xys, axis=1)
    b_1 = -k_1
    b_2 = 3 * k_1**2
    b_3 = -12 * k_1**3
    b_4 = 55 * k_1**4
    b_5 = -273 * k_1**5
    b_6 = 1428 * k_1**6
    b_7 = -7752 * k_1**7
    b_8 = 43263 * k_1**8
    b_9 = -246675 * k_1**9
    bs = [b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9]
    rsq = r**2
    rpow = rsq
    Q = 1
    for b in bs :
        Q = Q + b * rpow
        rpow = rpow * rsq
    return xys * Q.reshape(-1, 1)


# get camera intrinsics for pinhole model such that that the undistorted image
# contains the distorted image
def get_pinhole_from_simple_radial(camera_parameters) :
    points = np.array([[0, 0], 
                       [camera_parameters['resolution'][0] * 0.5 - 0.5, 0], 
                       [0, camera_parameters['resolution'][1] * 0.5 - 0.5]])
    points = points - camera_parameters['intrinsics'][:2, [2]].T
    points = points.dot(np.linalg.inv(camera_parameters['intrinsics'][:2, :2]))
    points = undistort_simple_radial(points, camera_parameters['distortion'])
    upper_left = points.min(axis=0)

    f = (- camera_parameters['intrinsics'][:2, [2]].T / upper_left).min()

    pinhole = {}
    pinhole['intrinsics'] = intrinsic_matrix(f, f, 
                                             camera_parameters['resolution'][0] * 0.5 - 0.5,
                                             camera_parameters['resolution'][1] * 0.5 - 0.5)
    pinhole['resolution'] = camera_parameters['resolution']
    
    return pinhole 


def set_visualizer_pose(visualizer, pose) :
    auto_params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = auto_params.intrinsic
    
    params.extrinsic = pose 
    
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(params)
    visualizer.poll_events()
    visualizer.update_renderer()


def set_visualizer_intrinsics(visualizer, intrinsics, resolution) :
    auto_params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = auto_params.intrinsic
    
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                           resolution[0],
                           resolution[1],
                           intrinsics[0, 0],
                           intrinsics[1, 1],
                           intrinsics[0, 2], 
                           intrinsics[1, 2])
    
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(params)
    visualizer.poll_events()
    visualizer.update_renderer()


# TODO: remove?
def project(X, camera, use_distortion=True) :
    X_transformed = X.dot(camera['R'].T) + camera['t']
    x = X_transformed[:, :2] / X_transformed[:, [2]]
    if use_distortion :
        x = distort_simple_radial(x, camera['distortion'])
    x = x.dot(camera['intrinsics'][:2, :2].T) + camera['intrinsics'][:2, 2]
    return x


def project_steps(X, pose, intrinsics, distortion=None) :
    X_C = X.dot(pose[:3, :3].T) + pose[:3, [3]].T
    x_C = X_C[:, :2] / X_C[:, [2]]
    if distortion is not None :
        x_Cd = distort_simple_radial(x_C, distortion)
        x_I = x_Cd.dot(intrinsics[:2, :2].T) + intrinsics[:2, 2]
        return x_I, x_Cd, x_C, X_C
    else :
        x_I = x_C.dot(intrinsics[:2, :2].T) + intrinsics[:2, 2]
        return x_I, x_C, X_C


# TODO: do we need to look over this again???
# TODO: look over this again
# TODO: return shape
def project_multiple_steps(X, poses, intrinsics, distortion=None) :
    X_repeat = X.reshape(1, -1, 3).repeat(len(poses), axis=0).reshape(len(poses), -1, 3)
    X_C = X_repeat @ poses[:, :3, :3].transpose(0, 2, 1) + translations.reshape(-1, 1, 3)
    # repeat
    x_C = X_C[:, :, :2] / X_C[:, :, [2]]
    if use_distortion : # TODO: different distortion parameters
        x_Cd = distort_simple_radial(x_C.reshape(-1, 2), distortion).reshape(-1, len(poses), 2)
        x_I = x_Cd @ intrinsics.reshape(-1, 3, 3)[:, :2, :2].transpose(0, 2, 1) + \
              intrinsics.reshape(-1, 3, 3)[:, :2, [2]].transpose(0, 2, 1)
        return x_I, x_Cd, x_C, X_C
    else :
        x_I = x_C @ intrinsics.reshape(-1, 3, 3)[:, :2, :2].transpose(0, 2, 1) + \
              intrinsics.reshape(-1, 3, 3)[:, :2, [2]].transpose(0, 2, 1)
        return x_I, x_C, X_C


# Project points into separate views
def project_separate_steps(X, poses, intrinsics, distortion=None) :
    shape = X.shape
    shape2d = X.shape[:-1] + (2,)
    
    poses = poses.reshape(-1, 4, 4)
    X = X.reshape(len(poses), -1, 3)
    
    X_C = X @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, :3, 3].reshape(-1, 1, 3)
    x_C = X_C[:, :, :2] / X_C[:, :, [2]]
    
    if distortion is not None : 
        x_Cd = distort_simple_radial(x_C.reshape(-1, 2), distortion).reshape(len(poses), -1, 2)
        x_I = x_Cd @ intrinsics[:2, :2].T.reshape(1, 2, 2) + intrinsics[:2, 2].reshape(-1, 1, 2)
        return x_I.reshape(shape2d), x_Cd.reshape(shape2d), x_C.reshape(shape2d), X_C.reshape(shape)
    else :
        x_I = x_C @ intrinsics[:2, :2].T.reshape(1, 2, 2) + intrinsics[:2, 2].reshape(-1, 1, 2)
        return x_I.reshape(shape2d), x_C.reshape(shape2d), X_C.reshape(shape)


# TODO: what intermediate step is this??
def jacobian_perspective(X_C) :
    J = np.zeros([len(X_C), 2, 3])
    J[:, 0, 0] = 1 / X_C[:, 2]
    J[:, 1, 1] = 1 / X_C[:, 2]
    J[:, 0, 2] = - X_C[:, 0] / (X_C[:, 2] * X_C[:, 2])
    J[:, 1, 2] = - X_C[:, 1] / (X_C[:, 2] * X_C[:, 2])
    return J


def jacobian_distortion(x_C, distortion) :
    J = np.zeros([len(x_C), 2, 2])
    rsq = x_C[:, 0]*x_C[:, 0] + x_C[:, 1]*x_C[:, 1]
    J[:, 0, 0] = (rsq + 2 * x_C[:, 0]* x_C[:, 0]) * distortion + 1
    J[:, 1, 1] = (rsq + 2 * x_C[:, 1]* x_C[:, 1]) * distortion + 1
    J[:, 0, 1] = x_C[:, 0] * x_C[:, 1] * 2 * distortion
    J[:, 1, 0] = x_C[:, 0] * x_C[:, 1] * 2 * distortion
    return J


def jacobian_projection(X_C, x_C, poses, intrinsics, distortion=None) :
    poses = poses.reshape(-1, 1, 4, 4)
    
    J_R = poses[:, :, :3, :3]
    
    J_pi = jacobian_perspective(X_C.reshape(-1, 3)).reshape(len(poses), -1, 2, 3)
    
    if distortion is not None :
        J_phi = jacobian_distortion(x_C.reshape(-1, 2), distortion).reshape(len(poses), -1, 2, 2)
    
    J_K = intrinsics[:2, :2].reshape(1, 1, 2, 2)
    
    if distortion is not None :
        return (J_K @ J_phi @ J_pi @ J_R).reshape(X_C.shape[:-1] + (2, 3))
    else :
        return (J_K @ J_pi @ J_R).reshape(X_C.shape[:-1] + (2, 3))


def get_grid_directions(points, normals) :
    horizontal = np.zeros(normals.shape)
    horizontal[:, 0] = -normals[:, 2]
    horizontal[:, 2] = normals[:, 0]
    horizontal = preprocessing.normalize(horizontal, axis=1)
    
    vertical = np.zeros(normals.shape)
    vertical = np.cross(normals, horizontal)
    vertical = preprocessing.normalize(vertical, axis=1)
    
    return horizontal, vertical


def create_grids(points, normals, scale=0.01) :
    horizontal, vertical = get_grid_directions(points, normals)
    grids = np.zeros([normals.shape[0], 16, 3])
    
    horizontal_scaling = np.array([-1., -1./3., 1./3., 1.]).reshape(1, 1, 4, 1) * scale.reshape(-1, 1, 1, 1)
    vertical_scaling = np.array([1., 1./3, -1./3, -1.]).reshape(1, 4, 1, 1) * scale.reshape(-1, 1, 1, 1)
    
    grids = points.reshape(-1, 1, 1, 3) + horizontal_scaling * horizontal.reshape(-1, 1, 1, 3) + \
                                          vertical_scaling * vertical.reshape(-1, 1, 1, 3)
    return grids


def optimize_grid_spacing_multiple(points, normals, poses, camera_parameters, visibility, iterations=10) :
    planes_horizontal, planes_vertical = get_grid_directions(points, normals)
    
    grid_scale = np.ones(len(points)) * 0.01
    grid_directions = np.zeros([len(points), 4, 3])
    grid_directions[:, 0, :] = planes_horizontal + planes_vertical
    grid_directions[:, 1, :] = planes_horizontal - planes_vertical
    grid_directions[:, 2, :] = -planes_horizontal - planes_vertical
    grid_directions[:, 3, :] = -planes_horizontal + planes_vertical
    grid_directions = preprocessing.normalize(grid_directions.reshape(-1, 3), axis=1) * 1.
    grid_directions = grid_directions.reshape(-1, 4, 3)
    
    visibility_idx = np.argwhere(visibility)
    
    for _ in range(iterations) :
        grid_outer = grid_scale.reshape(-1, 1, 1) * grid_directions + points.reshape(-1, 1, 3)
        
        # TODO: choose points and cameras
        poses_varrepeat = poses[visibility_idx[:, 1], :]
        x_I, _, x_C, X_C = project_separate_steps(grid_outer[visibility_idx[:, 0], :], 
                                                  poses_varrepeat, 
                                                  camera_parameters['intrinsics'], 
                                                  camera_parameters['distortion'])
        
        J = jacobian_projection(X_C, x_C, poses_varrepeat, 
                                camera_parameters['intrinsics'], 
                                camera_parameters['distortion'])
        
        distance_pair = np.zeros([x_I.shape[0], x_I.shape[1], 2, 2])
        distance_pair[:, 1:, 0, :] = x_I[:, :-1, :]
        distance_pair[:, 0, 0, :] = x_I[:, -1, :]
        distance_pair[:, :-1, 1, :] = x_I[:, 1:, :]
        distance_pair[:, -1, 1, :] = x_I[:, 0, :]
        
        d = np.zeros([distance_pair.shape[0], distance_pair.shape[1], 2])
        d[:, :, 0] = np.linalg.norm(x_I - distance_pair[:, :, 0, :], axis=2)
        d[:, :, 1] = np.linalg.norm(x_I - distance_pair[:, :, 1, :], axis=2)
        
        d_full = scipy.sparse.csr_matrix((d.reshape(-1, 8).mean(axis=1),
                                          (visibility_idx[:, 0], visibility_idx[:, 1])))
        d_means = np.asarray(d_full.sum(axis=1)).reshape(-1) / visibility.sum(axis=1)

        J_d = ((x_I - distance_pair[:, :, 0, :]) / d[:, :, 0, np.newaxis] + 
               (x_I - distance_pair[:, :, 1, :]) / d[:, :, 1, np.newaxis])
        
        gradient = J_d.reshape(-1, 4, 1, 2) @ J @ grid_directions[visibility_idx[:, 0]].reshape(-1, 4, 3, 1)
        
        #TODO: sparse matrix???
        g_full = scipy.sparse.csr_matrix((gradient.reshape(-1, 4).mean(axis=1), 
                                          (visibility_idx[:, 0], 
                                           visibility_idx[:, 1])))
        g = np.asarray(g_full.sum(axis=1)).reshape(-1) / visibility.sum(axis=1)

        grid_step = - 1. / g * (d_means - 3.)  # -(J^TJ)^-1Jr
        grid_scale = grid_scale + grid_step
        
        if (np.abs(d_means - 3.) < 1e-8).all() :
            break
    
    return grid_scale, grid_directions, grid_outer


    
    




"""
def optimize_grid_spacing(point, normal, cameras, iterations=50) :
    # 
    # TODO: jacobian    
    point = point.reshape(1, -1)
    
    plane_horizontal, plane_vertical = get_grid_directions(point.reshape(1, -1), 
                                                           normal.reshape(1, -1))
    plane_horizontal = plane_horizontal[0, :]
    plane_vertical = plane_vertical[0, :]
    
    
    grid_scale = 0.01
    grid_directions = np.zeros([4, 3])
    grid_directions[0, :] = plane_horizontal + plane_vertical
    grid_directions[1, :] = plane_horizontal - plane_vertical
    grid_directions[2, :] = -plane_horizontal - plane_vertical
    grid_directions[3, :] = -plane_horizontal + plane_vertical
    grid_directions = preprocessing.normalize(grid_directions, axis=1) * 1.
    
    first=False
    for iteration in range(iterations) :
        grid_outer = grid_scale * grid_directions + point
        
        x_I, _, x_C, X_C = project_multiple_steps(grid_outer, cameras, True)

        J = jacobian_projection(X_C, x_C, cameras, True)
        # x_I : views x points x 2
        #print(x_I)
        # TODO: redo for multiple images
        distance_pair = np.zeros([x_I.shape[0], x_I.shape[1], 2, 2])

        distance_pair[:, 1:, 0, :] = x_I[:, :-1, :]
        distance_pair[:, 0, 0, :] = x_I[:, -1, :]
        distance_pair[:, :-1, 1, :] = x_I[:, 1:, :]
        distance_pair[:, -1, 1, :] = x_I[:, 0, :]
        #print(distance_pair)
        # J_d = distance_jacobian(x_I, distance_pair) # TODO: reconsider
        
        d = np.zeros([distance_pair.shape[0], distance_pair.shape[1], 2])
        d[:, :, 0] = np.linalg.norm(x_I - distance_pair[:, :, 0, :], axis=2)
        d[:, :, 1] = np.linalg.norm(x_I - distance_pair[:, :, 1, :], axis=2)
        
        if first :
            grid_scale = 3 / d.mean() * grid_scale
            first=False
        else :
            # automize for arbitrarily many pairs # TODO: update reshape
            J_d = ((x_I - distance_pair[:, :, 0, :]) / d[:, :, 0, np.newaxis] + 
                   (x_I - distance_pair[:, :, 1, :]) / d[:, :, 1, np.newaxis]) #* \
                  #s(d.mean() - 3.).reshape(-1, 1)
            
            # TODO: change naming
            gradient = grid_directions.reshape(-1, 4, 1, 3) @ J @ J_d.reshape(-1, 4, 2, 1)
            
            gaussnewton_step = - 1. / gradient.mean() * (d.mean() - 3.)
            
            grid_scale = grid_scale + gaussnewton_step
            #grid_scale = grid_scale - 0.000004 * gradient.mean()
        if np.abs(d.mean() - 3.) < 1e-8 :
            break

    return grid_scale, grid_directions, grid_outer
"""