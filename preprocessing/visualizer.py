import numpy as np
from scipy import interpolate
import scipy.optimize

from PIL import Image
import matplotlib.pyplot as plt

import os.path
import time
import argparse

from utils import *
from robust_mean import loss, f, jac
import visualization
from gui import CloudGui

import colmap.read_write_model


parser = argparse.ArgumentParser()
parser.add_argument("--input-path", default="../../../data/training/ignatius_subset_workspace/optimization/preprocessing_low", type=str)


def main() :
    args = parser.parse_args()
    main_directory = args.input_path #"../../../data/training/ignatius_subset_workspace/optimization/"
    print(main_directory)
    cloud_names = ["initial",
                   "perturbed_initial",
                   "optimized",
                   "perturbed_optimized"]
    clouds = []
    for cloud_name in cloud_names :
        points_normals = np.loadtxt(os.path.join(main_directory, 
                                                 "points_normals_" + cloud_name + ".txt"))#[:, :6]
        print(points_normals.shape)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_normals[:, :3]))
        cloud.normals = o3d.utility.Vector3dVector(points_normals[:, 3:6])
        clouds = clouds + [cloud]
    
    user_interface = CloudGui(clouds, cloud_names)
    user_interface.run()
    
    


if __name__ == "__main__" :
    main()