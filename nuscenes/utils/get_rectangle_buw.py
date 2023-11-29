#from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import Box
import numpy as np
import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import cv2
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.utils.geometry_utils import view_points, transform_matrix

class RectanglesBUW():
    #def __init__(self, nusc: NuScenes, nuscex:NuScenesExplorer):
    #    self.nusc = nusc
    #    self.nuscex = nuscex

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
        new_box_x_max = max(corners.T[:,0]) # max of x-values
        new_box_x_min = min(corners.T[:,0])
        new_box_y_max = max(corners.T[:,1]) # max of y-values
        new_box_y_min = min(corners.T[:,1]) 

        return new_box_x_max, new_box_x_min, new_box_y_max, new_box_y_min
