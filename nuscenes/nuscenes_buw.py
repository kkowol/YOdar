import nuscenes
import numpy as np
from nuscenes.utils import data_classes
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image, ImageDraw
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from matplotlib.patches import Rectangle


class NuScenesBUW(NuScenes):
    ### inheritance from NuScenes devkit
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        #self.nuscex = nuscex

    def list_scenes_buw(self) -> None:
        desc_list, desc_list_compact, length_time_list, location_list, ann_count_list = self.explorer.list_scenes_buw()
        return desc_list, desc_list_compact, length_time_list, location_list, ann_count_list
    
    def get_tokens(self, sample_token: str, dot_size: int = 5, pointsensor_channel: str = 'LIDAR_TOP',
                                   camera_channel: str = 'CAM_FRONT', out_path: str = None) -> None:
        self.explorer.get_tokens(sample_token, dot_size, pointsensor_channel=pointsensor_channel,
                                                 camera_channel=camera_channel, out_path=out_path)

    def render_annotation_cam_front(self, sample_annotation_token: str,annotation: str, pref_class: str, margin: float = 10, view: np.ndarray = np.eye(4),
                          box_vis_level: BoxVisibility = BoxVisibility.ANY, plot_pics: bool = True,  out_path: str = None,
                          extra_info: bool = False) -> None:
        rectangle, annotation_name = self.explorer.render_annotation_cam_front(sample_annotation_token,annotation, pref_class, margin, view, box_vis_level, plot_pics, out_path, extra_info)
        return rectangle, annotation_name

    def render_scene_channel_pic(self, scene_token: str, channel: str = 'CAM_FRONT', freq: float = 10,
                             imsize: Tuple[float, float] = (640, 360), out_path: str = None) -> None:
        self.explorer.render_scene_channel_pic(scene_token, channel=channel, freq=freq, imsize=imsize, out_path=out_path)

    def list_scenes_buw(self) -> None:
        """ Lists all scenes with some meta data. """
        desc_list = []
        desc_list_compact = []
        length_time_list = []
        location_list = []
        ann_count_list = []

        def ann_count(record):
            count = 0
            sample = self.nusc.get('sample', record['first_sample_token'])
            while not sample['next'] == "":
                count += len(sample['anns'])
                sample = self.nusc.get('sample', sample['next'])
            return count

        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]

        for start_time, record in sorted(recs):
            start_time = self.nusc.get('sample', record['first_sample_token'])['timestamp'] / 1000000
            length_time = self.nusc.get('sample', record['last_sample_token'])['timestamp'] / 1000000 - start_time
            location = self.nusc.get('log', record['log_token'])['location']
            desc = record['name'] + ', ' + record['description']
            
            desc_list.append(desc) # get all informations about scene
            if len(desc) > 55: 
                desc = desc[:51] + "..."
                desc_list_compact.append(desc)
            if len(location) > 18:
                location = location[:18]

            print('{:16} [{}] {:4.0f}s, {}, #anns:{}'.format(
                desc, datetime.utcfromtimestamp(start_time).strftime('%y-%m-%d %H:%M:%S'),
                length_time, location, ann_count(record)))


            length_time_list.append(length_time)
            location_list.append(location)
            ann_count_list.append(ann_count(record))
        return desc_list, desc_list_compact, length_time_list, location_list, ann_count_list

    def get_tokens(self,
                   sample_token: str,
                   dot_size: int = 5,
                   pointsensor_channel: str = 'LIDAR_TOP',
                   camera_channel: str = 'CAM_FRONT',
                   out_path: str = None) -> None:                   
    
        sample_record = self.nusc.get('sample', sample_token)
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]
        scene_token = sample_record['scene_token']
        return pointsensor_token, camera_token, scene_token



    def render_annotation_cam_front(self,
                          anntoken: str,
                          annotation: str,
                          pref_class: str,
                          margin: float = 10,
                          view: np.ndarray = np.eye(4),
                          box_vis_level: BoxVisibility = BoxVisibility.ANY,
                          plot_pics: bool= True,
                          out_path: str = None,
                          extra_info: bool = False) -> None:
        """
        Render selected annotation.
        :param anntoken: Sample_annotation token.
        :param margin: How many meters in each direction to include in LIDAR view.
        :param view: LIDAR view point.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param out_path: Optional path to save the rendered figure to disk.
        :param extra_info: Whether to render extra information below camera view.
        """
        boxes_cam_front = []
        annotation_name = []
        cam_token = []
        list_anntoken = [] #distance#
        dict_anntoken = {} #distance#
        
        if plot_pics == True:
            fig, axes = plt.subplots(1, 1, figsize=(18, 9)) #for pic
        
        for i_run in range(len(anntoken)):
            if annotation[i_run] == pref_class:        # get annotations from preferred category, e.g. vehicle
                ann_record = self.nusc.get('sample_annotation', anntoken[i_run])
                sample_record = self.nusc.get('sample', ann_record['sample_token'])
                assert 'LIDAR_TOP' in sample_record['data'].keys(), 'No LIDAR_TOP in data, cant render'
        
                # Figure out which camera the object is fully visible in (this may return nothing)
                boxes, cam = [], []
                cam = 'CAM_FRONT'

                data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                            selected_anntokens=[anntoken[i_run]])
                if len(boxes) == 0:
                    continue
                else:
                    boxes_cam_front.append(boxes)
                    
                    cam_token.append(sample_record['data'][cam])
                    
                    # box size
                    rcParams['font.family'] = 'monospace'
                    w, l, h = ann_record['size']
                    category = ann_record['category_name']
                    lidar_points = ann_record['num_lidar_pts']
                    radar_points = ann_record['num_radar_pts']
        
                    sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                    pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
                    dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

                    #distance#
                    dict_anntoken['path'] = data_path
                    dict_anntoken['anntoken'] = anntoken[i_run]
                    dict_anntoken['distance'] = round(dist,3)
                    dict_anntoken['box_width'] = w
                    dict_anntoken['box_length'] = l
                    dict_anntoken['box_height'] = h
                    list_anntoken.append(dict_anntoken)
                    dict_anntoken = {}
        
        # Plot CAMERA view + boxes
        #'''    for pic
        if plot_pics == True:
            im = Image.open(data_path)
            axes.imshow(im)
            #axes.set_title(self.nusc.get('sample_data', cam)['channel'])
            axes.set_title('CAM_FRONT')
            axes.axis('off')
            axes.set_aspect('equal')
        #'''
        rectangle = []
        for box in boxes_cam_front:

            ''' #for 3D Boxes
            c = np.array(self.get_color(box[0].name)) / 255.0
            box[0].render(axes, view=camera_intrinsic, normalize=True, colors=(c, c, c))
            '''
            #''' for 2D Boxes
            #new_box_x_max, new_box_x_min, new_box_y_max, new_box_y_min = box[0].get_rectangle(view=camera_intrinsic, normalize=True)
            #test = box[0].render((view=camera_intrinsic, normalize=True))
            new_box_x_max, new_box_x_min, new_box_y_max, new_box_y_min = box[0].get_rectangle(view=camera_intrinsic, normalize=True)

            # crop boxes, if they leave the picture
            if new_box_x_max > 1600:
                new_box_x_max=1600
            if new_box_y_max > 900:
                new_box_y_max =900
            if new_box_x_min <0:
                new_box_x_min =0
            if new_box_y_min <0:
                new_box_y_min=0

            width = new_box_x_max - new_box_x_min
            height= new_box_y_max - new_box_y_min
            
            #for pic
            if plot_pics == True:
                axes.add_patch(Rectangle((new_box_x_min, new_box_y_min), width, height, 
                                     fill=None, edgecolor='orange', linewidth=2 ))
            
            rectangle.append([new_box_x_max, new_box_x_min, new_box_y_max, new_box_y_min],)
            #'''

        # Print extra information about the annotation below the camera view.
        if extra_info:
            for i_run2 in range(len(anntoken)):
                                    
                ann_record = self.nusc.get('sample_annotation', anntoken[i_run2])
                sample_record = self.nusc.get('sample', ann_record['sample_token'])
                
                rcParams['font.family'] = 'monospace'
    
                w, l, h = ann_record['size']
                category = ann_record['category_name']
                lidar_points = ann_record['num_lidar_pts']
                radar_points = ann_record['num_radar_pts']
    
                sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
                dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))
                
                print(str(i_run2) + ': ')
                print('distance: {:>7.3f}m'.format(dist))
                '''
                information = ' \n'.join(['category: {}'.format(category),
                                          '',
                                          '# lidar points: {0:>4}'.format(lidar_points),
                                          '# radar points: {0:>4}'.format(radar_points),
                                          '',
                                          'distance: {:>7.3f}m'.format(dist),
                                          '',
                                          'width:  {:>7.3f}m'.format(w),
                                          'length: {:>7.3f}m'.format(l),
                                          'height: {:>7.3f}m'.format(h)])
    
                plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                '''
        if out_path is not None:
            plt.savefig(out_path)
        '''
         # in case, that no annotation is available
        try:
            data_path
        except NameError:
            cam = 'CAM_FRONT'
            ann_record = self.nusc.get('sample_annotation', anntoken[0])
            sample_record = self.nusc.get('sample', ann_record['sample_token'])
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                            selected_anntokens=[anntoken[0]]) #first entry to get pic
        '''
        return rectangle, annotation_name, list_anntoken
    
    def render_instance(self,
                        instance_token: str,
                        margin: float = 10,
                        view: np.ndarray = np.eye(4),
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        out_path: str = None,
                        extra_info: bool = False) -> None:
        """
        Finds the annotation of the given instance that is closest to the vehicle, and then renders it.
        :param instance_token: The instance token.
        :param margin: How many meters in each direction to include in LIDAR view.
        :param view: LIDAR view point.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param out_path: Optional path to save the rendered figure to disk.
        :param extra_info: Whether to render extra information below camera view.
        """
        ann_tokens = self.nusc.field2token('sample_annotation', 'instance_token', instance_token)
        closest = [np.inf, None]
        for ann_token in ann_tokens:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            sample_record = self.nusc.get('sample', ann_record['sample_token'])
            sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
            dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))
            if dist < closest[0]:
                closest[0] = dist
                closest[1] = ann_token

        self.render_annotation(closest[1], margin, view, box_vis_level, out_path, extra_info)


    def render_scene_channel_pic(self,
                             scene_token: str,
                             channel: str = 'CAM_FRONT',
                             freq: float = 10,
                             imsize: Tuple[float, float] = (640, 360),
                             out_path: str = None) -> None:
        """
        Renders a full scene for a particular camera channel.
        :param scene_token: Unique identifier of scene to render.
        :param channel: Channel to render.
        :param freq: Display frequency (Hz).
        :param imsize: Size of image to render. The larger the slower this will run.
        :param out_path: Optional path to write a video file of the rendered frames.
        """

        valid_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        assert imsize[0] / imsize[1] == 16 / 9, "Aspect ratio should be 16/9."
        assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

        if out_path is not None:
            assert osp.splitext(out_path)[-1] == '.avi'

        # Get records from DB
        scene_rec = self.nusc.get('scene', scene_token)
        sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = self.nusc.get('sample_data', sample_rec['data'][channel])


        has_more_frames = True
        i_run = 0
        while has_more_frames:

            # Get data from DB
            impath, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_rec['token'],
                                                                        box_vis_level=BoxVisibility.ANY)

            # Load and render
            if not osp.exists(impath):
                raise Exception('Error: Missing image %s' % impath)
            im = cv2.imread(impath)
            for box in boxes:
                c = self.get_color(box.name)
                box.render_cv2(im, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Render
            cv2.imwrite('PATH/test_{}.png'.format(i_run), im)
            i_run += 1
            
            #if i_run >10:
             #   break
            if not sd_rec['next'] == "":
                sd_rec = self.nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False


    def get_matrix_radar(self, rectangle, im, points, coloring, slices, points_all): 
        """
        Parameters
        ----------
        rectangles : TYPE
            contains all points of the rectangles
        im : TYPE
            image of the scene
        points: TYPE
            contains points of the radar
        coloring: 
            information for depths
        points_all: Array
            0. x-axis (0-1600)
            1. y-axis (0-900)
            2. z-axis/coloring
            3. dyn_prop --> 8 possible states (0--> moving, 1--> stationary, 2 --> oncoming, 3--> stationary candidate, 
                                               4--> unknown, 5--> crossing stationary, 6--> crossing moving, 7--> stopped)
            4. id of the point
            5. rcs --> radar cross section (Radarquerschnitt) --> Rückstrahlfäche (echo area) 
            6. v_x
            7. v_y
            8. vx_comp --> ego velocity is considered
            9. vy_comp --> ego velocity is considered
            10. is_quality_valid
            11. ambig_state --> 5 possible states (0--> invalid, 1--> ambiguous, 2--> staggered ramp, 
                                                   3--> unambiguous, 4--> stationary candidates)
            12. x_rms
            13. y_rms
            14. invalid_state --> 18 possible states
            15. pdh0 --> 8 possible states 
            16. vx_rms
            17. vy_rms

        Returns
        -------
        points in rectangles/boxes

        """
        
        size_pic = im.size
        step = int(size_pic[0] / slices )
        
        matrix_data_preparation = np.zeros((12,slices))          #1./4 Änderung für Channels
        ''' Matrix_data_preparartion structure:
            0. id
            1. x-axis
            2. y-axis
            3. z-axis
            4. dyn_prop
            5. ambig_state
            6. vx
            7. vy
            8. place of the boxes on x-axis (binary)
            #9. hit (=1) or no hit (=0) in rectangle
            10.
        
        '''
        nr_slice = 0 
        for i_slice in range(0,size_pic[0], step): # run through all slices
            matrix_data_preparation[0,nr_slice]=nr_slice # write number of slice in first row
            for i_point in range(len(points_all[0,:])): # point number unequal to the number of slices!!!
                if points_all[0,:][i_point] >= i_slice and points_all[0,:][i_point] < i_slice+step:
                    
                    
                    # check if cell already exists, if so, then take the nearest point
                    if matrix_data_preparation[3,nr_slice] == 0:
                        matrix_data_preparation[1,nr_slice] = points_all[0,:][i_point] #x
                        matrix_data_preparation[2,nr_slice] = points_all[1,:][i_point] #y
                        matrix_data_preparation[3,nr_slice] = points_all[2,:][i_point] #z
                        matrix_data_preparation[4,nr_slice] = points_all[8,:][i_point] #vx_comp
                        matrix_data_preparation[5,nr_slice] = points_all[9,:][i_point] #vy_comp
                        matrix_data_preparation[6,nr_slice] = (points_all[3,:][i_point])+1 #dyn_prop +1 because 0 = moving
                        matrix_data_preparation[7,nr_slice] = points_all[11,:][i_point] #ambig_state
                        matrix_data_preparation[8,nr_slice] = points_all[14,:][i_point] #invalid_state
                        matrix_data_preparation[9,nr_slice] = points_all[5,:][i_point] #rcs (radar cross section, Radarquerschnitt = Rückstrahlfläche eines Objekts)
                        matrix_data_preparation[10,nr_slice] = points_all[15,:][i_point] #pdh0
                    elif matrix_data_preparation[3,nr_slice] > points_all[2,:][i_point]: # check if z is nearer
                        matrix_data_preparation[1,nr_slice] = points_all[0,:][i_point] #x
                        matrix_data_preparation[2,nr_slice] = points_all[1,:][i_point] #y
                        matrix_data_preparation[3,nr_slice] = points_all[2,:][i_point] #z
                        matrix_data_preparation[4,nr_slice] = points_all[8,:][i_point] #vx_comp
                        matrix_data_preparation[5,nr_slice] = points_all[9,:][i_point] #vy_comp
                        matrix_data_preparation[6,nr_slice] = (points_all[3,:][i_point])+1 #dyn_prop
                        matrix_data_preparation[7,nr_slice] = points_all[11,:][i_point] #ambig_state
                        matrix_data_preparation[8,nr_slice] = points_all[14,:][i_point] #invalid_state
                        matrix_data_preparation[9,nr_slice] = points_all[5,:][i_point] #rcs
                        matrix_data_preparation[10,nr_slice] = points_all[15,:][i_point] #pdh0
                    else:
                        continue
            nr_slice +=1    
                  
               
        '''
        ### find box in slices
        # Rectangle --> [0] --> x_max
        #               [1] -> x_min
        #               [2] -> y_max
        #               [3] -> y_min    
        '''
        matrix_rectangle = np.zeros((len(rectangle),slices)) # contains all boxes in picture, where every box has a row with ones and zeros
        
        ###################### added for velocity vectors #######################
        dyn_prop = matrix_data_preparation[6, :]
        velos = matrix_data_preparation[4, :]
        velo_prop = np.zeros((len(dyn_prop)))
        for i_prop in range(len(dyn_prop)):
            if dyn_prop[i_prop] == 1 or dyn_prop[i_prop] == 3 or dyn_prop[i_prop] == 7:
                velo_prop[i_prop] = velos[i_prop]
            else: 
                velo_prop[i_prop] = 0

        for i_rect in range(len(rectangle)):
            nr = 0
            for i_slice in range(0,size_pic[0], step): 
                
                # find range of min Box and max Box
                if i_slice >= rectangle[i_rect][1] and i_slice+step < rectangle[i_rect][0]:
                    matrix_rectangle[i_rect,nr] = 1
                    nr +=1
                else: 
                    matrix_rectangle[i_rect,nr] = 0
                    nr +=1

            mask = np.zeros( (slices), dtype=bool )# check if velocitiy in box
            for i_mask in range(slices):
                if dyn_prop[i_mask] == 1 or dyn_prop[i_mask] == 3 or dyn_prop[i_mask] == 7:
                    mask[i_mask] = True
                else:
                    mask[i_mask] = False
            
            if len(matrix_rectangle[i_rect][mask]) == 0:  # check if velocitiy in box, if not, delete row (set zeros)
                matrix_rectangle[i_rect, :] = 0
            
        ###################### added for velocity vectors #######################
        ### merge all rows in vector_rectangle
        #vector_rectangle= np.zeros((1,slices))
        if matrix_rectangle.shape[0] == 0:
            for i_rect in range(slices):
                matrix_data_preparation[11,i_rect]= 0            # 2./4 change
        else: 
            for i_rect in range(slices):
                test = np.max(matrix_rectangle[:,i_rect])       # 3./4 change
                if test >0:
                    matrix_data_preparation[11,i_rect]=1
                else:
                    matrix_data_preparation[11,i_rect]=0         # 4./4 change
 
        return matrix_data_preparation, matrix_rectangle

    def scene_selection(self,i_scene):
        """
        Parameters for mini data
        ----------
        i_scene : int
           scene number

        Returns
        -------
        scene : list 
            number of pictures in mini-dataset
        """

        if i_scene == 1:
            scene = [0,39]
        elif i_scene == 2:
            scene = [40, 80]
        elif i_scene == 3:
            scene = [81, 120]
        elif i_scene == 4:
            scene = [121, 161]
        elif i_scene == 5:
            scene = [162, 201]
        elif i_scene == 6:
            scene = [202, 241]
        elif i_scene == 7:
            scene = [242, 282]
        elif i_scene == 8:
            scene = [283, 323]
        elif i_scene == 9:
            scene = [324, 363]
        else:
            scene = [364, 404]
        return scene

    def map_pointcloud_to_image_buw(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar!'
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        points_rest = pc.points[3:,:]
        points_all = np.vstack([points, points_rest]) # merge all rows in matrix
        points_all[2, :] = coloring[:]   # colors saved in z
        point_id = pc.points[4,:]
        velo_x = pc.points[8, :]
        velo_y = pc.points[9, :]
        dyn_prop = pc.points[3, :]
        ambig_state = pc.points[11, :]
        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask] # delete invalid points
        points_all = points_all[:,mask]
        coloring = coloring[mask] # delete invalid points
        
        return points, coloring, im, points_all


    def map_pointcloud_sweep(self, 
                                    pointsensor_token: str, 
                                    camera_token: str, 
                                    current_sweeps: list, 
                                    scene: str, 
                                    pic: str, 
                                    time_sweeps: int, 
                                    path_sweeps: str, 
                                    min_dist: float = 1.0,
                                    render_intensity: bool = False) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param current_sweeps: List of unused sweeps in current scene and picture 
        :param scene: number of current scene
        :param pic: number of current picture in scene
        :param time_sweeps: number of sweeps before the current scene
        :param: path_sweeps: path of the sweeps folder
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token) # use the same pointsensor for the sweeps
        pcl_path = [osp.join(self.nusc.dataroot, pointsensor['filename'])]
        for i_path in range(-1,-(time_sweeps), -1): #get path of the sweeps (backwards)
            for f in os.listdir(path_sweeps):             
                if str(current_sweeps['timestamps_prev'][i_path]) in f:
                    pcl_path.append(osp.join(path_sweeps, f))
                    break # exit the second loop
        sweep_points = []
        sweep_coloring = []
        sweep_points_all = []
        for i_pc in range(len(pcl_path)):
            pc = RadarPointCloud.from_file(pcl_path[i_pc])
            im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))

            # Second step: transform to the global frame.
            poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pc.translate(np.array(poserecord['translation']))

            # Third step: transform into the ego vehicle frame for the timestamp of the image.
            poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
            pc.translate(-np.array(poserecord['translation']))
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

            # Fourth step: transform into the camera.
            cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            if render_intensity:
                assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar!'
                # Retrieve the color from the intensities.
                # Performs arbitary scaling to achieve more visually pleasing results.
                intensities = pc.points[3, :]
                intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
                intensities = intensities ** 0.1
                intensities = np.maximum(0, intensities - 0.5)
                coloring = intensities
            else:
                # Retrieve the color from the depth.
                coloring = depths

            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
            points_rest = pc.points[3:,:]
            points_all = np.vstack([points, points_rest]) # merge all rows to matrix
            points_all[2, :] = coloring[:] 
            point_id = pc.points[4,:]
            velo_x = pc.points[8, :]
            velo_y = pc.points[9, :]
            dyn_prop = pc.points[3, :]
            ambig_state = pc.points[11, :]
            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
            points = points[:, mask] # delete invalid points
            points_all = points_all[:,mask]
            coloring = coloring[mask] # delete invalid points
            sweep_points.append(points)
            sweep_coloring.append(coloring)
            sweep_points_all.append(points_all)

        return sweep_points, sweep_coloring, im, sweep_points_all

    
    def get_radar_tokens(self, 
                pointsensor_token:str, 
                camera_token, min_dist: float = 1.0,
                render_intensity: bool = False) -> Tuple:
        
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])

        if pointsensor['sensor_modality'] == 'lidar':
            print('This file is no radar data')
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

        
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

    
    def get_distance(self, anntoken):
        for i_run in range(len(anntoken)):
            ann_record = self.nusc.get('sample_annotation', anntoken[i_run])
            sample_record = self.nusc.get('sample', ann_record['sample_token'])
            rcParams['font.family'] = 'monospace'
            w, l, h = ann_record['size']
            category = ann_record['category_name']
            lidar_points = ann_record['num_lidar_pts']
            radar_points = ann_record['num_radar_pts']
            sample_data_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
            dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))
            
            print(str(i_run) + ': ')
            print('distance: {:>7.3f}m'.format(dist))


    def get_rectangle(self,
               #axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        
                       bb1        bb2              nb1             nb2
                        x----------x                x---------------x
                     -          -  -                -               -
             bf1 x----------x bf2  -                -               -
                 -          -      -        -->     -               -
                 -     bb3  -      -                -               -
                 -      x   -      x bb4            -               -
                 -          -   -                   -               -
             bf3 x----------x bf4                   x---------------x
                                                   nb3             nb4
        box_front_1 = corners.T[0,:]
        box_front_2 = corners.T[1,:]
        box_front_3 = corners.T[2,:]
        box_front_4 = corners.T[3,:]
        box_back_1 = corners.T[4,:]
        box_back_2 = corners.T[5,:]
        box_back_3 = corners.T[6,:]
        box_back_4 = corners.T[7,:] 
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        #create new box/rectangle
        new_box_x_max = max(corners.T[:,0]) # Maximum der X-Werte
        new_box_x_min = min(corners.T[:,0])
        new_box_y_max = max(corners.T[:,1]) # Maximum der Y-Werte
        new_box_y_min = min(corners.T[:,1]) 

        return new_box_x_max, new_box_x_min, new_box_y_max, new_box_y_min

    
    def render_bird_view(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        """
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                NuScenesExplorer.render_ego_centric_map(self, sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(NuScenesExplorer.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # show area of interest
            ax.add_patch(Rectangle((0, 0), 5, 5, 
                                     fill=True, edgecolor='orange', linewidth=2 ))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            #ax.set_xlim(0, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == 'camera':
            raise ValueError("Error: Not implemented yet!")

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title(sd_record['channel'])
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path)
    

    def get_angle(self, 
                       sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       axes_limit: float = 40,
                       ax: Axes = None,
                       use_flat_vehicle_coordinates: bool = True) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        """
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            #ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

        # Get boxes in lidar frame.
        _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)
        
        # box_angles list of tuple: (degree, annotation, class)
        box_angles = [] 
        for box in boxes:
            box_angles.append((box.orientation.degrees, box.token, box.name))
        
        return box_angles
