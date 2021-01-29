import numpy as np
from scipy import interpolate
import scipy.optimize

import open3d as o3d
from PIL import Image

import os.path
import time

from utils import *
from robust_mean import loss

import colmap.read_write_model


input_path = "../../../data/horse_workspace/"
input_format = ".txt"

cameras, images, points3D = colmap.read_write_model.read_model(path=input_path,
                                                        ext=input_format)


dense_cloud = o3d.io.read_point_cloud(input_path + "dense/0/fused.ply")
dense_cloud.normalize_normals()
mesh = o3d.io.read_triangle_mesh(input_path + "dense/0/meshed-poisson.ply")

points = np.array(dense_cloud.points)
normals = np.array(dense_cloud.normals)
colors = np.array(dense_cloud.colors)

# TODO: determine correct number of cameras from
camera_dicts = np.array([get_camera_parameters(cameras[idx], images[idx]) for idx in range(1, 149)])

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080, visible=False)
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()

images = [Image.open(input_path + "images/" + '0' * (5 - len(str(i))) + str(i) + ".jpg") for i in range(1, 150)]
depth_images = []

for camera in camera_dicts :
    set_visualizer_camera_parameters(vis, camera)
    depth_images.append(vis.capture_depth_float_buffer())


x = np.arange(0, 1920, 1)
y = np.arange(0, 1080, 1)

recompute_visibility = False
if os.path.isfile("visibility_matrix.npy") and not recompute_visibility :
    visibility_matrix = np.load("visibility_matrix.npy")
    print("Loaded visibility matrix")
else :
    visibility_matrix = np.zeros([len(camera_dicts), len(points)], dtype=bool)

    margin = 5

    for i, camera in enumerate(camera_dicts) :
        print("Camera " + str(i))
        x_image, _, X_C = project_steps(points, camera, False)
        point_selector = np.logical_and(
                             np.logical_and(x_image[:, 0] >= 0+margin, x_image[:, 1] >= 0+margin),
                             np.logical_and(x_image[:, 0] < 1920+margin, x_image[:, 1] < 1080+margin))
        x_image = x_image[point_selector, :]
        X_C = X_C[point_selector, :]
        depth_image_bilin = interpolate.RectBivariateSpline(y, x, np.array(depth_images[i]), kx=1, ky=1)
        ratio = depth_image_bilin(x_image[:, 1], x_image[:, 0], grid=False) / X_C[:, 2]
        
        depth_selection = np.logical_and(ratio < 1.01, ratio > 0.99)
        depth_selection = np.logical_and(depth_selection, X_C[:, 2] < 4.)
        
        point_selector[point_selector] = depth_selection
        visibility_matrix[i, :] = point_selector

    np.save("visibility_matrix", visibility_matrix)
    



visibility_matrix = visibility_matrix.T
visible_points = visibility_matrix.sum(axis=1) > 2
print(visible_points.sum())
visibility_matrix = visibility_matrix[visible_points, :]
points = points[visible_points, :]
normals = normals[visible_points, :]




print("start")

z = np.arange(0, len(images), 1)
image_stack = np.array([np.array(image) for image in images]).mean(axis=3)
images_interp = interpolate.RegularGridInterpolator((z, y, x), image_stack)


batch_size = 10000
secs_all = time.time()
secs = time.time()


stacked_grid_spacing = True

if stacked_grid_spacing :
    scales, _, grid_outer = optimize_grid_spacing_multiple(points[:batch_size, :], normals[:batch_size, :], camera_dicts, visibility_matrix[:batch_size, :])
    scales = scales / np.sqrt(2)
else :
    for idx in range(batch_size) :
        if idx % 500 == 0 : 
            print(idx)
        visible_idx = np.argwhere(visibility_matrix[idx, :]).reshape(-1)
        visible_cameras = [camera_dicts[i] for i in visible_idx]
        
        # optimize grid spacing
        grid_scale, grid_directions, grid_outer = \
            optimize_grid_spacing(points[[idx], :], normals[[idx], :], 
                                  visible_cameras)


print("finished optimize grid spacing")
print(time.time() - secs)

visibility_idx = np.argwhere(visibility_matrix[:batch_size, :])

grids = create_grids(points[:batch_size], normals[:batch_size], scales)

grids_I, _, _, _ = project_separate_steps(grids[visibility_idx[:, 0]].reshape(-1, 16, 3), camera_dicts[visibility_idx[:, 1]], True)



# TODO: plot 





grid_z = visibility_idx[:, [1]].repeat(16, axis=1)

grids_I_interp_coord = np.concatenate([grid_z.reshape(-1, 1), 
                                       np.flip(grids_I.reshape(-1, 2), axis=1)], axis=1)

secs = time.time()
patch_stack = images_interp(grids_I_interp_coord).reshape(-1, 16)

patch_stack_un = patch_stack
patch_stack_un_norm = np.linalg.norm(patch_stack_un, axis=1)
print("Filtering")
print(patch_stack_un_norm.min())


# TODO: normalize patches
mu = patch_stack.mean(axis=1).reshape(-1, 1)
sigma = np.linalg.norm(patch_stack - mu, axis=1).reshape(-1, 1)

patch_stack = (patch_stack - mu) / sigma

# TODO: how to transfer this validity
valid = (sigma!=0).reshape(-1)
patch_stack = patch_stack[valid, :]


visibility_matrix_local = visibility_matrix[:batch_size, :]
visibility_idx = np.argwhere(visibility_matrix_local)
visibility_matrix_local[visibility_idx[:, 0], visibility_idx[:, 1]] = (sigma != 0).reshape(-1)
visibility_idx = np.argwhere(visibility_matrix_local)

secs = time.time()
#TODO: sparse matrix # TODO: list of 16 sparse matrices
patch_fulls = [scipy.sparse.csr_matrix((patch_stack[:, i] , (visibility_idx[:, 0], visibility_idx[:, 1]))) for i in range(16)]
print("patch_fulls")
patch_mean = np.zeros([visibility_matrix_local.shape[0], 16])
for i, patch_full in enumerate(patch_fulls) :
    patch_mean[:, i] = (np.asarray(patch_full.sum(axis=1)).reshape(-1) / visibility_matrix_local.sum(axis=1))

#patch_full = np.zeros([visibility_matrix_local.shape[0], len(camera_dicts), 16])
#patch_full[:] = np.nan
#patch_full[visibility_idx[:, 0], visibility_idx[:, 1], :] = patch_stack
#patch_mean = np.nanmean(patch_full, axis=1)

print(patch_mean.shape)

print("success")
print(time.time() - secs)

stacked_optimization = True

if stacked_optimization :
# Robust Mean Calculation
    row_idx = np.array(range(len(visibility_idx[:, 0]))).reshape(-1, 1).repeat(16, 1).reshape(-1)
    col_idx = np.array(range(batch_size*16)).reshape(-1, 16)[visibility_idx[:, 0], :]
    col_idx = col_idx.reshape(-1)

    def f(x, patches, visibility_idx) :
        residuals = np.linalg.norm(x.reshape(-1, 16)[visibility_idx[:, 0], :] - patches, axis=1)
        return residuals
    def jac(x, patches, visibility_idx) :
        # TODO: sparse matrix
        vals = x.reshape(-1,16)[visibility_idx[:, 0], :] / np.linalg.norm(x.reshape(-1, 16)[visibility_idx[:, 0], :] - patches, axis=1).reshape(-1, 1)
        print(vals.shape)
        j = scipy.sparse.csr_matrix((vals.reshape(-1), (row_idx, col_idx)))
        return j
        
    # Matrix to indicate sparsity of the jacobian
    jac_sparsity = scipy.sparse.csr_matrix((np.ones(len(visibility_idx[:, 0])*16), 
                                           (row_idx, col_idx)))
    secs = time.time()
    result = scipy.optimize.least_squares(f, x0=patch_mean.reshape(-1), jac=jac,
                                loss=loss, args=(patch_stack, visibility_idx), jac_sparsity=jac_sparsity,
                                method='trf')
else :
    def f(x, patch) :
        return np.linalg.norm(x.reshape(-1, 16) - patch.reshape(-1, 16), axis=1)
    def jac(x, patch) :
        return x.reshape(1, -1) / np.linalg.norm(x.reshape(-1, 16) - patch.reshape(-1, 16), axis=1).reshape(-1, 1)
    secs = time.time()
    for i in range(len(patch_mean)) :
        if i % 500 == 0:
            print(i)
        result = scipy.optimize.least_squares(f, x0=patch_mean[i, :], jac=jac,
                                loss=loss, args=(patch_stack[visibility_idx[i,0], :].reshape(1, -1, 16)),
                                method='dogbox')
        if i==0 :
            print(result.x)
print(result)

print("finished optimize least squares")
print(time.time() - secs)
robust_mean = result.x.reshape(-1, 16)
print(time.time() - secs_all)
print("End")

