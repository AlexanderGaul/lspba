import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import matplotlib.colors as pltcolors

from utils import get_views_center
from visualization import get_camera_symbols

class CropGui :
    def __init__(self, point_cloud, poses, camera_parameters) :
        self.point_cloud = point_cloud
        self.poses = poses
        self.camera_parameters = camera_parameters
        
        self.initial_location = get_views_center(poses)

        # Transformation parameters for bounding box
        self.rotation = np.array([0., 0., 0.])
        self.location = self.initial_location
        self.scale = np.array([3., 3., 3.])
        
        self.member_box = self.get_transformed_box

        # Rotation widgets
        self.crop_button = gui.Button("Crop")
        
        self.rot_x_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.rot_y_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.rot_z_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.rot_x_slider.set_limits(-180., 180.)
        self.rot_y_slider.set_limits(-180., 180.)
        self.rot_z_slider.set_limits(-180., 180.)

        self.loc_x_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.loc_y_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.loc_z_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.loc_x_slider.set_limits(-5.+self.location[0], 5.+self.location[0])
        self.loc_y_slider.set_limits(-5.+self.location[1], 5.+self.location[1])
        self.loc_z_slider.set_limits(-5.+self.location[2], 5.+self.location[2])
        self.loc_x_slider.double_value = self.location[0]
        self.loc_y_slider.double_value = self.location[1]
        self.loc_z_slider.double_value = self.location[2]
        
        self.scale_x_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.scale_y_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.scale_z_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self.scale_x_slider.set_limits(0.01, 10.)
        self.scale_y_slider.set_limits(0.01, 10.)
        self.scale_z_slider.set_limits(0.01, 10.)
        self.scale_x_slider.double_value = 3.0
        self.scale_y_slider.double_value = 3.0
        self.scale_z_slider.double_value = 3.0
        
        
        self.rot_label = gui.Label("Rotation")
        self.loc_label = gui.Label("Locations")
        self.scale_label = gui.Label("Scaling")
        
        
        self.options = gui.Vert()
        self.options.add_child(self.crop_button)
        self.options.add_child(self.rot_label)
        self.options.add_child(self.rot_x_slider)
        self.options.add_child(self.rot_y_slider)
        self.options.add_child(self.rot_z_slider)
        self.options.add_child(self.loc_label)
        self.options.add_child(self.loc_x_slider)
        self.options.add_child(self.loc_y_slider)
        self.options.add_child(self.loc_z_slider)
        self.options.add_child(self.scale_label)
        self.options.add_child(self.scale_x_slider)
        self.options.add_child(self.scale_y_slider)
        self.options.add_child(self.scale_z_slider)
        
        # Set event functions
        self.crop_button.set_on_clicked(self._on_crop)
        self.rot_x_slider.set_on_value_changed(self._on_rot_x)
        self.rot_y_slider.set_on_value_changed(self._on_rot_y)
        self.rot_z_slider.set_on_value_changed(self._on_rot_z)
        self.loc_x_slider.set_on_value_changed(self._on_loc_x)
        self.loc_y_slider.set_on_value_changed(self._on_loc_y)
        self.loc_z_slider.set_on_value_changed(self._on_loc_z)
        self.scale_x_slider.set_on_value_changed(self._on_scale_x)
        self.scale_y_slider.set_on_value_changed(self._on_scale_y)
        self.scale_z_slider.set_on_value_changed(self._on_scale_z)
        
        # Scene set-up
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Landmark Selection", 1920, 1080)
        
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.enable_sun_light(True)
        self.scene.visible = True
        
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.options)
        
        box = self.get_transformed_box()
           
        # TODO: what is this for
        self.mat = rendering.Material()
        self.mat.point_size = 100
        self.mat_box = rendering.Material()
        self.mat_box.line_width = 4.
        #mat_box.thickness = 4.
        #mat_box.point_size = 4.
        #mat_box.shader = o3d.visualization.O3DVisualizer.Shader.UNLIT
        #mat_box.base_color = [255, 0, 0, 0]
        box.color = np.array([1, 0, 0])
        
        self.scene.scene.add_geometry("cloud", self.point_cloud, self.mat)
        self.scene.scene.add_geometry("box", box, self.mat_box)

        self.camera_symbols = get_camera_symbols(self.poses, self.camera_parameters)
        for i, symbol in enumerate(self.camera_symbols) :
            self.scene.scene.add_geometry("camera_"+str(i), symbol, self.mat_box)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.scene.setup_camera(60, bbox, [0, 0, 0])
        
        

    def _on_layout(self, theme) :
        r = self.window.content_rect
        self.scene.frame = gui.Rect(r.x, r.y, r.width-400, r.height)
        self.options.frame = gui.Rect(r.width-400, r.y, 400, r.height)
    
    def _on_crop(self) :
        self.scene.scene.remove_geometry("cloud")
        self.scene.scene.add_geometry("cloud", self.point_cloud.crop(self.get_transformed_box()), self.mat)
            
    def _on_rot_x(self, x) :
        self.rotation[0] = x
        self._transform_box()
    def _on_rot_y(self, y) :
        self.rotation[1] = y
        self._transform_box()
    def _on_rot_z(self, z) :
        self.rotation[2] = z
        self._transform_box()
    
    def _on_loc_x(self, x) :
        self.location[0] = x
        self._transform_box()
    def _on_loc_y(self, y) :
        self.location[1] = y
        self._transform_box()
    def _on_loc_z(self, z) :
        self.location[2] = z
        self._transform_box()
        
    def _on_scale_x(self, x) :
        self.scale[0] = x
        self._transform_box()
    def _on_scale_y(self, y) :
        self.scale[1] = y
        self._transform_box()
    def _on_scale_z(self, z) :
        self.scale[2] = z
        self._transform_box()
    
    def _transform_box(self) :
        box_T = self.get_transformed_box()
        box_T.color = np.array([1, 0, 0])
        self.scene.scene.remove_geometry("box")
        self.scene.scene.add_geometry("box", box_T, self.mat)

    def get_transformed_box(self) :
        extreme_points = np.array([self.scale, -self.scale])
        box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(extreme_points))
        box = box.get_oriented_bounding_box()

        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_yzx(self.rotation / 360. * 3.14)
        box_T = o3d.geometry.OrientedBoundingBox(box)
        box_T = box_T.rotate(R)
        box_T = box_T.translate(self.location)
        return box_T

    def get_selection_box_parameters(self):
        return self.rotation, self.location, self.scale
        
    def run(self):
        gui.Application.instance.run()
    
 

class CloudGui :
    def __init__(self, point_clouds, names) :
        self.point_clouds = point_clouds
        self.line_sets = []
        self.names = names
        self.color_selection = [pltcolors.to_rgb('tab:blue'), pltcolors.to_rgb('tab:orange'),
                                pltcolors.to_rgb('tab:green'), pltcolors.to_rgb('tab:red')]
        
        # TODO create line set for normals
        
        self.normals = []
        for i, cloud in enumerate(self.point_clouds) :
            points2 = np.asarray(cloud.points) - np.asarray(cloud.normals) * 0.02
            pointsline = np.concatenate([points2, np.asarray(cloud.points)], axis=0)
            idx1 = np.arange(0, len(points2), 1)
            idx2 = idx1 + len(points2)
            lineidxs = np.concatenate([idx1.reshape(-1, 1), idx2.reshape(-1, 1)], 1)
            
            lineset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pointsline), 
                                           o3d.utility.Vector2iVector(lineidxs))
            
            colors = np.array(self.color_selection[i]).reshape(1, -1).repeat(len(cloud.points), 0)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            lineset.colors = cloud.colors
            
            self.line_sets = self.line_sets + [lineset]
        
        
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Landmark Selection", 1920, 1080)
        
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.enable_sun_light(True)
        self.scene.visible = True
        
        self.mat = rendering.Material()
        
        self.check_boxes = []
        for i, _ in enumerate(self.point_clouds) :
            self.check_boxes = self.check_boxes + [gui.Checkbox(self.names[i])]
        
        self.options = gui.Vert()
        for check_box in self.check_boxes :
            self.options.add_child(check_box)

        for i, check_box in enumerate(self.check_boxes) :
            check_box.set_on_checked(self.Check(self, i))
        
        # Scene set-up
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.options)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5],
                                                   [5, 5, 5])
        self.scene.setup_camera(60, bbox, [0, 0, 0])
    
    class Check :
        def __init__(self, gui, i) :
            self.i = i
            self.gui = gui
        
        def __call__(self, on) :
            if on :
                self.gui.scene.scene.add_geometry("cloud_"+str(self.i), self.gui.point_clouds[self.i], self.gui.mat)
                self.gui.scene.scene.add_geometry("lines_"+str(self.i), self.gui.line_sets[self.i], self.gui.mat)
            else :
                self.gui.scene.scene.remove_geometry("cloud_"+str(self.i))
                self.gui.scene.scene.remove_geometry("lines_"+str(self.i))

    def _on_layout(self, theme) :
        r = self.window.content_rect
        self.scene.frame = gui.Rect(r.x, r.y, r.width-400, r.height)
        self.options.frame = gui.Rect(r.width-400, r.y, 400, r.height)
    
    def run(self):
        gui.Application.instance.run()