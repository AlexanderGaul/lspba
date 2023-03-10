import numpy as np
from scipy import interpolate
import scipy.optimize

from PIL import Image
import matplotlib.pyplot as plt

import os.path
import time

from utils import *
from robust_mean import loss, f, jac
import visualization
from gui import CropGui

import colmap.read_write_model

# TODO: command line arguments


input_path = "../../../data/family_workspace/"
input_format = ".bin"

output_path = input_path + "preprocessing/"

model_cameras, model_images, _ = \
    colmap.read_write_model.read_model(path=input_path+"colmap/sparse/0/", ext=input_format)

dense_cloud = o3d.io.read_point_cloud(input_path + "colmap/dense/0/fused.ply")
dense_cloud.normalize_normals()
mesh = o3d.io.read_triangle_mesh(input_path + "colmap/dense/0/meshed-poisson.ply")

points = np.array(dense_cloud.points)
normals = np.array(dense_cloud.normals)
colors = np.array(dense_cloud.colors)

# Extract poses and camera parameters from model variables
poses = np.array([get_pose(model_images[key]) for key in model_images.keys()])
# TODO: handle radial distortion with two parameters
camera_parameters = get_camera_parameters(model_cameras[1])
intrinsics = camera_parameters['intrinsics'] 
distortion = camera_parameters['distortion']
width, height = camera_parameters['resolution']


# save poses and camera intrinsics
with open(os.path.join(input_path, "preprocessing/camera.txt"), "w+") as file:
    np.savetxt(file, np.array([intrinsics[0, 0], intrinsics[1, 1], 
                               intrinsics[0, 2], intrinsics[1, 2],
                               distortion, 0]).reshape(1, -1))
with open(os.path.join(input_path, "preprocessing/poses.txt"), "w+") as file:
    np.savetxt(file, np.concatenate([poses[:, :3, :3].reshape(-1, 9), 
                                     poses[:, :3, 3].reshape(-1, 3)], axis=1))


# TODO: when to visualize
show_viewer = True
if show_viewer :
    user_interface = CropGui(dense_cloud, poses, camera_parameters)
    user_interface.run()
    
    cropped_cloud = dense_cloud.crop(user_interface.get_transformed_box())
    points = np.array(cropped_cloud.points)
    normals = np.array(cropped_cloud.normals)
    colors = np.array(cropped_cloud.colors)
    
    
    


recompute_visibility = True
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
        
        print(i)
        print(points[12111])
        print(x_image[12111, :])
        
        visible_selection = np.logical_and(visible_depth, visible_image)
        
        x_depth = x_depth[visible_selection, :]
        x_image = x_image[visible_selection, :]
        X_C =  X_C[visible_selection, :]
        normals_selection = normals[visible_selection, :]
        
        depth_image_bilin = interpolate.RegularGridInterpolator((np.arange(0, height, 1),
                                                                 np.arange(0, width, 1)), 
                                                                np.array(depth_images[i]))
        
        ratio = depth_image_bilin(np.flip(x_depth, axis=1)) / X_C[:, 2]
        depth_selection = np.logical_and(ratio < 1.01, ratio > 0.99)

        # Only select close points, optional
        depth_selection = np.logical_and(depth_selection, X_C[:, 2] < 4.)
        
        #depth_selection = np.logical_and(depth_selection, 
        #                                 np.arccos(normals_selection.dot(pose[[2], :3].T)).reshape(-1) > (np.pi / 2. + np.pi / 4))
        
        visible_selection[visible_selection] = depth_selection
        
        visibility_matrix[:, i] = visible_selection

    np.save("visibility_matrix", visibility_matrix)


points_indices = np.arange(0, len(points), 1)

# Remove points that are not visible
visible_points = visibility_matrix.sum(axis=1) > 0
visibility_matrix = visibility_matrix[visible_points, :]
points_all = points

points = points[visible_points, :]
normals = normals[visible_points, :]
colors = colors[visible_points, :]

indx_visible = points_indices[visible_points]


# Load images and create interpolation function
images = [Image.open(input_path + "images/" + '0' * (5 - len(str(i))) + str(i) + ".jpg") for i in range(1, len(poses)+1)]
image_stack = np.array([np.array(image) for image in images]).mean(axis=3)
images_interp = interpolate.RegularGridInterpolator((np.arange(0, len(images), 1), 
                                                     np.arange(0, height, 1), 
                                                     np.arange(0, width, 1)), 
                                                    image_stack,
                                                    bounds_error=False, fill_value=None)

batch_size = 10000
landmarks_found = 0

# TODO: batch index to continue

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
    
    grids = create_grids(points[batch_begin:batch_end], normals[batch_begin:batch_end], scales)
    grids_I, _, _, _ = project_separate_steps(grids[visibility_idx[:, 0]].reshape(-1, 16, 3), 
                                              poses[visibility_idx[:, 1], :], intrinsics, distortion)
    grid_z = visibility_idx[:, [1]].repeat(16, axis=1)
    grids_I_interp_coord = np.concatenate([grid_z.reshape(-1, 1), 
                                           np.flip(grids_I.reshape(-1, 2), axis=1)], axis=1)

    patch_stack = images_interp(grids_I_interp_coord).reshape(-1, 16)
    
    # Normalize patches
    mu = patch_stack.mean(axis=1).reshape(-1, 1)
    # TODO: discard invalid sigmas right here
    patch_stack_centered = patch_stack - mu
    sigma = np.linalg.norm(patch_stack_centered, axis=1).reshape(-1, 1)
    patch_stack = (patch_stack_centered) / sigma
    
    # TODO: rename  ??
    valid_patch = (sigma!=0).reshape(-1)
    patch_stack = patch_stack[valid_patch, :]
    sigma = sigma[valid_patch]
    grids_I = grids_I[valid_patch, :]
    
    visibility_matrix_batch = visibility_matrix[batch_begin:batch_end, :]
    visibility_idx = np.argwhere(visibility_matrix_batch)
    visibility_matrix_batch[visibility_idx[:, 0], visibility_idx[:, 1]] = valid_patch
    visibility_idx = np.argwhere(visibility_matrix_batch)
    # TODO: how do landmark indices change here if at all # indx_local = indx_visible[batch_begin:batch_end][]
    # TODO: remove rows without visibility

    secs = time.time()
    patch_fulls = [scipy.sparse.csr_matrix((patch_stack[:, i] , (visibility_idx[:, 0], visibility_idx[:, 1]))) for i in range(16)]
    patch_mean = np.zeros([visibility_matrix_batch.shape[0], 16])
    for i, patch_full in enumerate(patch_fulls) :
        patch_mean[:, i] = (np.asarray(patch_full.sum(axis=1)).reshape(-1) / visibility_matrix_batch.sum(axis=1))
    
    patch_full = np.zeros([visibility_matrix_batch.shape[0], len(poses), 16])
    patch_full[:] = np.nan
    patch_full[visibility_idx[:, 0], visibility_idx[:, 1], :] = patch_stack
    patch_median = np.nanmedian(patch_full, axis=1)

    robust_mean = np.zeros(patch_mean.shape)
    
    
    for i in range(len(patch_mean)) :
        if not (visibility_idx[:,0]==i).any() : continue # TODO: test 
        result = scipy.optimize.least_squares(f, x0=patch_mean[i, :], jac=jac, 
                                loss=loss, args=(patch_stack[visibility_idx[:,0]==i, :].reshape(1, -1, 16)),
                                method='dogbox', ftol=1e-4, xtol=1e-4, gtol=1e-4, verbose=0) #, tr_solver='lsmr', verbose=0)
        robust_mean[i] = result.x
        
    print("Optimize robust mean: " + str(time.time() - secs))
    
    # Determine valid landmarks that are visible
    # TODO: do this before computing even the patch mean??
    num_visibilities = visibility_matrix_batch.sum(axis=1)
    valid_points = num_visibilities > 0
    
    # Determine source views
    distances = np.linalg.norm(patch_stack - robust_mean[visibility_idx[:, 0], :].reshape(-1, 16), axis=1)
    D = scipy.sparse.csr_matrix((-(distances-distances.max()), (visibility_idx[:, 0], visibility_idx[:, 1])))
    source_idx = np.asarray(D.argmax(axis=1)[valid_points]).reshape(-1)
    
    # Determine landmarks that have source patches with sufficient texture
    source_idx_repeat = source_idx.repeat(num_visibilities[valid_points])
    enough_texture = sigma[visibility_idx[:, 1] == source_idx_repeat].reshape(-1) > 8.
    valid_points[valid_points] = enough_texture
    source_idx = source_idx[enough_texture]
    
    
    # Append batch results to files
    
    with open(os.path.join(input_path, "preprocessing/points_normals_gridscales_select.txt"), 
              "w+" if batch_idx == 0 else "a") as file:
        np.savetxt(file, np.concatenate([points[batch_begin:batch_end, :][valid_points, :], 
                                      normals[batch_begin:batch_end, :][valid_points, :], 
                                      scales.reshape(-1, 1)[valid_points, :]], axis=1))
    
    ### TODO: remove source index from visibility indices
    ### TODO: shift indices according to previously determined landmarks
    ### TODO: truncate matrix
    visibility_matrix_batch_valid = visibility_matrix_batch[valid_points, :]
    visibility_idx = np.argwhere(visibility_matrix_batch_valid)
    with open(os.path.join(input_path, "preprocessing/visibility.txt"), 
              "w+" if batch_idx == 0 else "a") as file :
        np.savetxt(file, visibility_idx + np.array([landmarks_found, 0]), '%d')
    
    # Compute landmarks
    points_C = points[batch_begin:batch_end, :][valid_points, :].reshape(-1, 1, 3) \
             @ poses[source_idx, :3, :3].transpose(0, 2, 1) + \
               poses[source_idx, :3, 3].reshape(-1, 1, 3)
    normals_C = normals[batch_begin:batch_end, :][valid_points, :].reshape(-1, 1, 3) @ poses[source_idx, :3, :3].transpose(0, 2, 1)
    n = get_plane(points_C.reshape(-1, 3), 
                  normals_C.reshape(-1, 3))

    x, _, _, _ = project_separate_steps(points[batch_begin:batch_end, :][valid_points, :], 
                                        poses[source_idx, :], 
                                        intrinsics, distortion)

    with open(os.path.join(input_path, "preprocessing/landmarks.txt"), 
              "w+" if batch_idx == 0 else "a") as file :
        np.savetxt(file, np.concatenate([x, n], axis=1))
    
    with open(os.path.join(input_path, "preprocessing/source_views.txt"), 
              "w+" if batch_idx == 0 else "a") as file :
        np.savetxt(file, source_idx, '%d')
    
    landmarks_found += valid_points.sum()
    
    print("Results for batch " + str(batch_idx) + " written")
    
    # TODO: print runtime for whole batch
    
    
    if False :
        plot_idx = 10
        idxs = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        for landmark_idx in idxs :
            print(landmark_idx)
            if not (indx_visible == landmark_idx).any() :
                continue
            plot_idx = np.argwhere(indx_visible == landmark_idx).reshape(-1)[0]
            print(plot_idx)
            visualization.plot_grids(grids_I[visibility_idx[:, 0] == plot_idx, :, :],
                                     visibility_idx[visibility_idx[:, 0] == plot_idx, 1], 
                                     np.array([np.array(image) for image in images]))
            visualization.plot_patches(patch_stack[visibility_idx[:, 0] == plot_idx, :], 
                                       robust_mean[plot_idx, :], patch_median[plot_idx, :], patch_mean[plot_idx, :], 
                                       visibility_idx[visibility_idx[:, 0] == plot_idx, 1])
            plt.show()
    

