import numpy as np
import matplotlib.pyplot as plt

from utils import *

def plot_grids(grid_I, visibility_idx, images, area=40) :
    # TODO
    grid_centers = grid_I.mean(axis=1)

    plot_height = int(np.sqrt(len(grid_I)))
    plot_width = int(np.ceil(len(grid_I) / plot_height))

    fig = plt.figure()
    current_plot = 1
    for i, idx in enumerate(visibility_idx) :
        lower = grid_centers[i,:].astype(int) - area
        higher = grid_centers[i, :].astype(int) + area
        higher = higher - np.minimum(lower, 0)
        bounds = np.array([np.array(images[idx]).shape[1], np.array(images[idx]).shape[0]])
        lower = lower - np.maximum(higher, bounds) + bounds
        lower = np.maximum(lower, 0)
        higher = np.minimum(higher, bounds)
        
        ax = fig.add_subplot(plot_height, plot_width, current_plot)
        ax.imshow(np.array(images[idx])[lower[1]:higher[1], lower[0]:higher[0]], interpolation="bilinear")
        # TODO: 
        ax.scatter(grid_I[i, :, 0] - lower[0], grid_I[i, :, 1] - lower[1],
                   facecolors='none', edgecolors='r', linewidth=0.5, s=2)
        # TODO: change title
        ax.set_title("View " + str(idx), fontsize=8)
        ax.set_axis_off()
        current_plot += 1
        #ax.set_yticklabels([]); ax.set_xticklabels([])
    #plt.show()



def plot_patches(patches_stack, robust_mean, median, mean, view_idxs=None) :
    # TODO
    num_images = len(patches_stack)
    vmax = 0.6
    
    closest = np.argmin(np.linalg.norm(patches_stack - robust_mean.reshape(-1, 16), axis=1))
    # TODO: find closest
    
    plot_height = int(max(2, np.sqrt(num_images) + 1))
    plot_width = int(max(4, np.ceil(num_images / (plot_height - 1))))
    current_plot = 1
    
    fig = plt.figure()
    
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(robust_mean.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Robust Mean", fontsize=8)
    current_plot += 1
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(median.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Median", fontsize=8)
    current_plot += 1
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(mean.reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Mean", fontsize=8)
    current_plot += 1
    ax = fig.add_subplot(plot_height, plot_width, current_plot)
    ax.imshow(patches_stack[closest, :].reshape(4,4), vmin=-vmax, vmax=vmax)
    ax.set_axis_off()
    ax.set_title("Closest View " + str(view_idxs[closest]), fontsize=8)
    current_plot += 1
    current_plot = plot_width+1
    for i in range(len(patches_stack)) :
        ax = fig.add_subplot(plot_height, plot_width, current_plot)
        ax.imshow(patches_stack[i, :].reshape(4,4), vmin=-vmax, vmax=vmax)
        ax.set_axis_off()
        ax.set_title("View " + str(view_idxs[i]), fontsize=8)
        current_plot += 1
    
    #plt.show()


def combined_plot(images, patches_stack, robust_mean, mean, closest) :
    pass



def get_camera_symbols(poses, camera_parameters) :
    centers, _, _, _ = get_centers_directions(poses)
    
    #points = np.zroes([centers.shape[0], 5, 3])
    #points[:, 0, :] = centers
    
    length = 0.1
    # TODO: unproject corners
    points = np.array([[0, 0], 
                       [0, camera_parameters['resolution'][1]], 
                       camera_parameters['resolution'],
                       [camera_parameters['resolution'][0], 0]])
    points = (points - camera_parameters['intrinsics'][:2, [2]].T).dot(
             np.linalg.inv(camera_parameters['intrinsics'])[:2, :2].T)
    points = undistort_simple_radial(points, camera_parameters['distortion'])
    points = np.concatenate([points, np.ones(4).reshape(-1, 1)], axis=1) * 0.2
    points = poses[:, :3, :3].reshape(-1, 1, 3, 3).transpose(0, 1, 3, 2) @ \
             (points.reshape(1, 4, 3, 1) - poses[:, :3, 3].reshape(-1, 1, 3, 1))
    points = np.concatenate([centers.reshape(-1, 1, 3, 1),
                             points.reshape(-1, 4, 3, 1)], axis=1).reshape(-1, 5, 3)
    #print(points)
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], 
                      [1, 2], [2, 3], [3, 4], [4, 1]])
    colors = np.array([[1, 0, 0]]).repeat(8, axis=0)
    symbols = []
    for i in range(points.shape[0]) :
        line_set = o3d.geometry.LineSet()
        
        line_set.points = o3d.utility.Vector3dVector(points[i, :])
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        symbols = symbols + [line_set]
        
    return symbols
