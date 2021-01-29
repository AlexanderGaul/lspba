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
    
# TODO: define data type for camera parameters
def get_camera_parameters(camera_data, image_data) :
    camera_parameters = {}
    pose = get_pose(image_data)
    camera_parameters['R'] = pose[:3, :3]
    camera_parameters['t'] = pose[:3, 3]
    camera_parameters['distortion'] = camera_data.params[3]
    camera_parameters['intrinsics'] = intrinsic_matrix(camera_data.params[0],
                                                       camera_data.params[0],
                                                       camera_data.params[1],
                                                       camera_data.params[2])
    camera_parameters['width'] = camera_data.width
    camera_parameters['height'] = camera_data.height
    return camera_parameters

def distort_simple_radial(xys, k_1) :
    r = np.linalg.norm(xys, axis=1)
    return xys * (1 + k_1 * r * r)[:, np.newaxis]

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

def get_grid_directions(points, normals) :
    horizontal = np.zeros(normals.shape)
    horizontal[:, 0] = -normals[:, 2]
    horizontal[:, 2] = normals[:, 0]
    horizontal = preprocessing.normalize(horizontal, axis=1)
    
    vertical = np.zeros(normals.shape)
    vertical = np.cross(normals, horizontal)
    vertical = preprocessing.normalize(vertical, axis=1)
    
    return plane_horizontal, plane_vertical


def create_grids(points, normals, scale=0.01) :
    horizontal, vertical = get_grid_directions(points, normals)
    grids = np.zeros([normals.shape[0], 16, 3])
    
    # horizontal negative, vertical positive -> horizontal positive, vertical negative
    # horizontal: [[-1., -1./3., 1./3., 1.]]
    # vertical: [[1.], [1./3.], [-1./3.], [-1.]]
    horizontal_scaling = np.array([-1., -1./3., 1./3., 1.]).reshape(1, 1, 4, 1) * scale.reshape(-1, 1, 1, 1)
    vertical_scaling = np.array([1., 1./3, -1./3, -1.]).reshape(1, 4, 1, 1) * scale.reshape(-1, 1, 1, 1)
    
    grids = points.reshape(-1, 1, 1, 3) + horizontal_scaling * horizontal.reshape(-1, 1, 1, 3) + \
                                          vertical_scaling * vertical.reshape(-1, 1, 1, 3)
    return grids


# Sets camera parameters for Open3D visualizer
def set_visualizer_camera_parameters(visualizer, camera_dict) :
    # TODO: decode camera parameters
    auto_params = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = auto_params.intrinsic
    
    # Open3D only support pinhole camera intrinsics
    # Open3D expects pixel centers to be at 0.5
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                           camera_dict['width'],
                           camera_dict['height'],
                           camera_dict['intrinsics'][0, 0],
                           camera_dict['intrinsics'][1, 1],
                           camera_dict['width'] / 2 - 0.5, 
                           camera_dict['height'] / 2 - 0.5)
    # TODO: figure out 0.5 or 0
    pose = np.eye(4)
    pose[:3, :3] = camera_dict['R']
    pose[:3, 3] = camera_dict['t']
    params.extrinsic = pose 

    visualizer.get_view_control().convert_from_pinhole_camera_parameters(params)
    visualizer.poll_events()
    visualizer.update_renderer()


# TODO: rename ??
def project(X, camera, use_distortion=True) :
    X_transformed = X.dot(camera['R'].T) + camera['t']
    x = X_transformed[:, :2] / X_transformed[:, [2]]
    if use_distortion :
        x = distort_simple_radial(x, camera['distortion'])
    x = x.dot(camera['intrinsics'][:2, :2].T) + camera['intrinsics'][:2, 2]
    return x


def project_steps(X, camera, use_distortion=True) :
    X_C = X.dot(camera['R'].T) + camera['t']
    x_C = X_C[:, :2] / X_C[:, [2]]
    if use_distortion :
        x_Cd = distort_simple_radial(x_C, camera['distortion'])
        x_I = x_Cd.dot(camera['intrinsics'][:2, :2].T) + camera['intrinsics'][:2, 2]
        return x_I, x_Cd, x_C, X_C
    else :
        x_I = x_C.dot(camera['intrinsics'][:2, :2].T) + camera['intrinsics'][:2, 2]
        return x_I, x_C, X_C


# TODO: return shape
def project_multiple_steps(X, cameras, use_distortion=True) :
    # X : N x 3
    # cameras M
    N, _ = X.shape
    rotations = np.array([camera['R'].T for camera in cameras])
    translations = np.array([camera['t'].T for camera in cameras])
    
    intrinsics = np.array([camera['intrinsics'] for camera in cameras])
    
    X_repeat = X.reshape(1, -1, 3).repeat(len(rotations), axis=0).reshape(len(rotations), -1, 3)
    X_C = X_repeat @ rotations.reshape(-1, 3, 3) + translations.reshape(-1, 1, 3)
    # repeat
    x_C = X_C[:, :, :2] / X_C[:, :, [2]]
    if use_distortion : # TODO: different distortion parameters
        x_Cd = distort_simple_radial(x_C.reshape(-1, 2), cameras[0]['distortion']).reshape(-1, N, 2)
        x_I = x_Cd @ intrinsics.reshape(-1, 3, 3)[:, :2, :2].transpose(0, 2, 1) + \
              intrinsics.reshape(-1, 3, 3)[:, :2, [2]].transpose(0, 2, 1)
        return x_I, x_Cd, x_C, X_C
    else :
        x_I = x_C @ intrinsics.reshape(-1, 3, 3)[:, :2, :2].transpose(0, 2, 1) + \
              intrinsics.reshape(-1, 3, 3)[:, :2, [2]].transpose(0, 2, 1)
        return x_I, x_C, X_C


# Project points into separate cameras
def project_separate_steps(X, cameras, use_distortion=True) :
    # X : N x M x 3
    # cameras N
    # M is subgroup
    N, M, _ = X.shape
    rotations = np.array([camera['R'].T for camera in cameras])
    translations = np.array([camera['t'].T for camera in cameras])
    intrinsics = np.array([camera['intrinsics'] for camera in cameras])
    
    X_C = X.reshape(N, M, 1, 3) @ rotations.reshape(N, 1, 3, 3) + translations.reshape(N, 1, 1, 3)
    X_C = X_C.reshape(N, M, 3)
    # repeat
    x_C = X_C[:, :, :2] / X_C[:, :, [2]]
    if use_distortion : # TODO: different distortion parameters
        x_Cd = distort_simple_radial(x_C.reshape(-1, 2), cameras[0]['distortion']).reshape(N, M, 2)
        x_I = x_Cd.reshape(N, M, 1, 2) @ intrinsics[:, :2, :2].transpose(0, 2, 1).reshape(N, 1, 2, 2) + \
              intrinsics[:, :2, [2]].transpose(0, 2, 1).reshape(N, 1, 1, 2)
        return x_I.reshape(N, M, 2), x_Cd, x_C, X_C
    else :
        # TODO: check this
        x_I = x_C @ intrinsics.reshape(-1, 3, 3)[:, :2, :2].transpose(0, 2, 1) + \
              intrinsics.reshape(-1, 3, 3)[:, :2, [2]].transpose(0, 2, 1)
        return x_I, x_C, X_C

# TODO: define undistort??
def unproject(x, z, camera, use_distortion=False) :
    x_dist = (x - camera['intrinsics'][:2, 2]).dot(np.linalg.inv(camera['intrinsics'][:2, :2]).T)
    x_un = undistort(x_dist, camera['distortion'])
    X = np.ones(x_un.shape[0], 3)
    X = X * z.reshape(-1, 1)
    X = (X - camera['t']).dot(camera['R'])
    return X


# TODO: naming of functions
# Jac_perspective
# jac_distortion

# TODO: rename X_C
def project_jacobian(X) :
    J = np.zeros([len(X), 3, 2])
    J[:, 0, 0] = 1 / X[:, 2]
    J[:, 1, 1] = 1 / X[:, 2]
    J[:, 2, 0] = - X[:, 0] / (X[:, 2] * X[:, 2])
    J[:, 2, 1] = - X[:, 1] / (X[:, 2] * X[:, 2])
    return J

def distort_jacobian(x, dist) :
    J = np.zeros([len(x), 2, 2])
    rsq = x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1]
    J[:, 0, 0] = (rsq + 2 * x[:, 0]* x[:, 0]) * dist + 1
    J[:, 1, 1] = (rsq + 2 * x[:, 1]* x[:, 1]) * dist + 1
    J[:, 1, 0] = x[:, 0] * x[:, 1] * 2 * dist
    J[:, 0, 1] = x[:, 0] * x[:, 1] * 2 * dist
    return J




# TODO: single camera
def full_project_jacobian(X_C, x_C, camera_dict, use_distortion=True) :
    
    R_block = camera_dict['R'].T.reshape(1, 3, 3).repeat(len(X_C), axis=0)
    
    J_pi = project_jacobian(X_C)
    # TODO: small X
    J_phi = distort_jacobian(x_C, camera_dict['distortion'])
    
    In_block = camera_dict['intrinsics'][:2, :2].T.reshape(1, 2, 2).repeat(len(X_C), axis=0)
    
    J = R_block @ J_pi @ J_phi @ In_block
    if not use_distortion :
        J = R_block @ J_pi @ In_block
    
    return J

# TODO calculate intermediate coordinates inside the function itself
def full_project_jacobian_batch(X_C, x_C, cameras, use_distortion=True) :
    # X_C : N x M x 3
    # x_C
    # returns J : N x N x 3 x 2
    
    N, M, _ = X_C.shape
    #print("--------------")
    # stack poses
    # poses : N x 1 x 3 x 3
    R_stacked = np.array([camera['R'].T for camera in cameras])
    R_stacked = R_stacked.reshape(N, 1, 3, 3)
    #print(R_stacked)
    # jacobian for projection
    # J_pi : N x M x 3 x 2
    J_pi = project_jacobian(X_C.reshape(-1, 3))    # J_pi should be NM x 3 x 2
    J_pi = J_pi.reshape(N, M, 3, 2)
    #print(J_pi)
    # jacobian for distortion
    # J_phi : N x M x 2 x 2
    J_phi = distort_jacobian(x_C.reshape(-1, 2), cameras[0]['distortion'])    # TODO: generalize for different 
    J_phi = J_phi.reshape(N, M, 2, 2)
    #print(J_phi)
    # jacobian for pixel
    # 1 x 1 x 2 x 2
    F_stacked = np.array([camera['intrinsics'][:2, :2].T for camera in cameras])
    F_stacked = F_stacked.reshape(-1, 1, 2, 2)
    #print(F_stacked)
    if use_distortion :
        J = R_stacked @ J_pi @ J_phi @ F_stacked
    else :
        J = R_stacked @ J_pi @ F_stacked
    return J


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

        J = full_project_jacobian_batch(X_C, x_C, cameras, True)
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



def optimize_grid_spacing_multiple(points, normals, cameras, visibility, iterations=10) :
    # TODO: how to define cameras
    # visibility : points x cameras
    print(visibility.shape)
    print("degree of sparsity")
    print(visibility.sum() / len(points))
    cameras = np.array(cameras)
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
        print("iteration")
        secs = time.time()
        grid_outer = grid_scale.reshape(-1, 1, 1) * grid_directions + points.reshape(-1, 1, 3)
        print(time.time()-  secs)
        secs = time.time()
        # TODO: choose points and cameras
        cameras_varrepeat = cameras[visibility_idx[:, 1]]
        x_I, _, x_C, X_C = project_separate_steps(grid_outer[visibility_idx[:, 0], :], cameras_varrepeat, True)
        # cameras x allpoints x 3
        print(time.time()-  secs)
        secs = time.time()
        J = full_project_jacobian_batch(X_C, x_C, cameras_varrepeat, True)
        print(time.time() - secs)
        secs = time.time()
        distance_pair = np.zeros([x_I.shape[0], x_I.shape[1], 2, 2])
        distance_pair[:, 1:, 0, :] = x_I[:, :-1, :]
        distance_pair[:, 0, 0, :] = x_I[:, -1, :]
        distance_pair[:, :-1, 1, :] = x_I[:, 1:, :]
        distance_pair[:, -1, 1, :] = x_I[:, 0, :]
        
        d = np.zeros([distance_pair.shape[0], distance_pair.shape[1], 2])
        d[:, :, 0] = np.linalg.norm(x_I - distance_pair[:, :, 0, :], axis=2)
        d[:, :, 1] = np.linalg.norm(x_I - distance_pair[:, :, 1, :], axis=2)
        
        
        # TODO: replace with sparse
        d_full = scipy.sparse.csr_matrix((d.reshape(-1, 8).mean(axis=1),
                                          (visibility_idx[:, 0], visibility_idx[:, 1])))
        d_means = np.asarray(d_full.sum(axis=1)).reshape(-1) / visibility.sum(axis=1)
        print(time.time() - secs)
        secs = time.time()
        J_d = ((x_I - distance_pair[:, :, 0, :]) / d[:, :, 0, np.newaxis] + 
               (x_I - distance_pair[:, :, 1, :]) / d[:, :, 1, np.newaxis])
        
        gradient = grid_directions[visibility_idx[:, 0]].reshape(-1, 4, 1, 3) @ J @ J_d.reshape(-1, 4, 2, 1)
        print(time.time() - secs)
        secs = time.time()
        #TODO: sparse matrix???
        g_full = scipy.sparse.csr_matrix((gradient.reshape(-1, 4).mean(axis=1), 
                                          (visibility_idx[:, 0], 
                                           visibility_idx[:, 1])))
        g = np.asarray(g_full.sum(axis=1)).reshape(-1) / visibility.sum(axis=1)
        print(time.time() - secs)
        grid_step = - 1. / g * (d_means - 3.)  # -(J^TJ)^-1Jr
        grid_scale = grid_scale + grid_step

        if (np.abs(d_means - 3.) < 1e-8).all() :
            break
    
    return grid_scale, grid_directions, grid_outer


    
    



