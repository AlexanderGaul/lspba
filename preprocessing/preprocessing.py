import numpy as np
from scipy import interpolate
import scipy.optimize

import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt

import os.path
import time

from utils import *
from robust_mean import loss
import visualization

import colmap.read_write_model


input_path = "../../../data/horse_workspace/"
input_format = ".txt"

model_cameras, model_images, _ = \
    colmap.read_write_model.read_model(path=input_path, ext=input_format)

dense_cloud = o3d.io.read_point_cloud(input_path + "dense/0/fused.ply")
dense_cloud.normalize_normals()
mesh = o3d.io.read_triangle_mesh(input_path + "dense/0/meshed-poisson.ply")

points = np.array(dense_cloud.points)
normals = np.array(dense_cloud.normals)
colors = np.array(dense_cloud.colors)

# Extract poses and camera parameters from model variables
poses = np.array([get_pose(model_images[key]) for key in model_images.keys()])
camera_parameters = get_camera_parameters(model_cameras[1])
intrinsics = camera_parameters['intrinsics'] 
distortion = camera_parameters['distortion']
width, height = camera_parameters['resolution']


recompute_visibility = False
if os.path.isfile("visibility_matrix.npy") and not recompute_visibility :
    visibility_matrix = np.load("visibility_matrix.npy")
    print("Loaded visibility matrix")
else :
    visibility_matrix = np.zeros([len(points), len(poses)], dtype=bool)
    # Render depth images from mesh
    pinhole = get_pinhole_from_simple_radial(camera_parameters)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    set_visualizer_intrinsics(vis, pinhole['intrinsics'], pinhole['resolution'])
    
    depth_images = []
    for pose in poses :
        set_visualizer_pose(vis, pose)
        depth_images.append(vis.capture_depth_float_buffer())
    
    margin = 5
    
    for i, pose in enumerate(poses) :
        if i % 10 == 0 :
            print("Visibility for view " + str(i))
        x_depth, a, X_C = project_steps(points, pose, pinhole['intrinsics'])
        x_image, b, c, d = project_steps(points, pose, intrinsics, distortion)
        
        visible_depth = np.logical_and(np.logical_and(x_depth[:, 0] >= 0, 
                                                      x_depth[:, 1] >= 0),
                                       np.logical_and(x_depth[:, 0] < width, 
                                                      x_depth[:, 1] < height))
        
        visible_image = np.logical_and(np.logical_and(x_image[:, 0] >= 0 + margin, 
                                                      x_image[:, 1] >= 0 + margin),
                                       np.logical_and(x_image[:, 0] < width - margin, 
                                                      x_image[:, 1] < height - margin))
        
        visible_selection = np.logical_and(visible_depth, visible_image)
        
        x_depth = x_depth[visible_selection, :]
        x_image = x_image[visible_selection, :]
        X_C =  X_C[visible_selection, :]
        
        depth_image_bilin = interpolate.RegularGridInterpolator((np.arange(0, height, 1),
                                                                 np.arange(0, width, 1)), 
                                                                np.array(depth_images[i]))
        
        ratio = depth_image_bilin(np.flip(x_depth, axis=1)) / X_C[:, 2]
        depth_selection = np.logical_and(ratio < 1.01, ratio > 0.99)

        # Only select close points, optional
        depth_selection = np.logical_and(depth_selection, X_C[:, 2] < 4.)
        
        visible_selection[visible_selection] = depth_selection
        
        visibility_matrix[:, i] = visible_selection

    np.save("visibility_matrix", visibility_matrix)


# Remove points that are not visible
visible_points = visibility_matrix.sum(axis=1) > 0
visibility_matrix = visibility_matrix[visible_points, :]
points = points[visible_points, :]
normals = normals[visible_points, :]
colors = colors[visible_points, :]


# Load images and create interpolation function
images = [Image.open(input_path + "images/" + '0' * (5 - len(str(i))) + str(i) + ".jpg") for i in range(1, len(poses)+1)]
image_stack = np.array([np.array(image) for image in images]).mean(axis=3)
images_interp = interpolate.RegularGridInterpolator((np.arange(0, len(images), 1), 
                                                     np.arange(0, height, 1), 
                                                     np.arange(0, width, 1)), 
                                                    image_stack,
                                                    bounds_error=False)


batch_size = 10000

print(len(points))
print(range(int(np.ceil(len(points) / batch_size))))
for batch_idx in range(int(np.ceil(len(points) / batch_size))) :
    batch_begin = batch_idx*batch_size
    batch_end = (batch_idx+1)*batch_size
    secs = time.time()
    
    scales, _, _ = optimize_grid_spacing_multiple(points[batch_begin:batch_end, :], 
                                                  normals[batch_begin:batch_end, :], 
                                                  poses,
                                                  camera_parameters, 
                                                  visibility_matrix[batch_begin:batch_end, :])
    scales = scales / np.sqrt(2)
    print("Optimize grid spacing: " + str(time.time() - secs))


    visibility_idx = np.argwhere(visibility_matrix[batch_begin:batch_end, :])

    grids = create_grids(points[:batch_size], normals[batch_begin:batch_end], scales)
    grids_I, _, _, _ = project_separate_steps(grids[visibility_idx[:, 0]].reshape(-1, 16, 3), 
                                              poses[visibility_idx[:, 1], :], intrinsics, distortion)


    grid_z = visibility_idx[:, [1]].repeat(16, axis=1)
    grids_I_interp_coord = np.concatenate([grid_z.reshape(-1, 1), 
                                           np.flip(grids_I.reshape(-1, 2), axis=1)], axis=1)
    patch_stack = images_interp(grids_I_interp_coord).reshape(-1, 16)


    mu = patch_stack.mean(axis=1).reshape(-1, 1)
    sigma = np.linalg.norm(patch_stack - mu, axis=1).reshape(-1, 1)
    patch_stack = (patch_stack - mu) / sigma
    
    valid = (sigma!=0).reshape(-1)
    patch_stack = patch_stack[valid, :]
    grids_I = grids_I[valid, :]
    
    visibility_matrix_local = visibility_matrix[batch_begin:batch_end, :]
    visibility_idx = np.argwhere(visibility_matrix_local)
    visibility_matrix_local[visibility_idx[:, 0], visibility_idx[:, 1]] = (sigma != 0).reshape(-1)
    visibility_idx = np.argwhere(visibility_matrix_local)
    
    patch_fulls = [scipy.sparse.csr_matrix((patch_stack[:, i] , (visibility_idx[:, 0], visibility_idx[:, 1]))) for i in range(16)]
    
    patch_mean = np.zeros([visibility_matrix_local.shape[0], 16])
    patch_min = np.zeros([visibility_matrix_local.shape[0], 16])
    patch_max = np.zeros([visibility_matrix_local.shape[0], 16])
    
    for i, patch_full in enumerate(patch_fulls) :
        patch_mean[:, i] = (np.asarray(patch_full.sum(axis=1)).reshape(-1) / visibility_matrix_local.sum(axis=1))
        patch_min[:, i] = patch_full.min(axis=1).todense().reshape(-1)
        patch_max[:, i] = patch_full.max(axis=1).todense().reshape(-1)
    
    
    patch_full = np.zeros([visibility_matrix_local.shape[0], len(poses), 16])
    patch_full[:] = np.nan
    patch_full[visibility_idx[:, 0], visibility_idx[:, 1], :] = patch_stack
    patch_median = np.nanmedian(patch_full, axis=1)

    robust_mean = np.zeros(patch_mean.shape)
    
    def f(x, patch) :
        residual = np.linalg.norm(x.reshape(-1, 16) - patch.reshape(-1, 16), axis=1)
        return residual
    def jac(x, patch) :
        j = (x.reshape(-1, 16) - patch.reshape(-1, 16)) / np.linalg.norm(x.reshape(-1, 16) - patch.reshape(-1, 16), axis=1).reshape(-1, 1)
        return j
    
    secs = time.time()
    for i in range(len(patch_mean)) :
        result = scipy.optimize.least_squares(f, x0=patch_mean[i, :], jac=jac, 
                                loss=loss, args=(patch_stack[visibility_idx[:,0]==i, :].reshape(1, -1, 16)),
                                method='dogbox') #, tr_solver='lsmr', verbose=0)
        robust_mean[i] = result.x
        
    print("Optimize robust mean: " + str(time.time() - secs))


    plot_idx = 10
    for plot_idx in [100, 200, 300, 400, 500] :
        visualization.plot_grids(grids_I[visibility_idx[:, 0] == plot_idx, :, :],
                                 visibility_idx[visibility_idx[:, 0] == plot_idx, 1], 
                                 np.array([np.array(image) for image in images]))
        visualization.plot_patches(patch_stack[visibility_idx[:, 0] == plot_idx, :], 
                                   robust_mean[plot_idx, :], patch_median[plot_idx, :], patch_mean[plot_idx, :], 
                                   visibility_idx[visibility_idx[:, 0] == plot_idx, 1])
        plt.show()
    

