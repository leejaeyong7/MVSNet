import logging
import numpy as np
from PIL import Image
from os import path

from nvm_camera import NVMCamera
from nvm_point import NVMPoint

from j_math.point import Point
from j_math.rotation import Rotation

def is_int(str):
    '''Checks if string is integer'''
    try:
        int(str)
        return True
    except:
        return False

class NVMModel:
    '''Generic parser for NVM file

    This file will parse NVM model from NVM file string
    This file will contain a list of NVMCamera, and a list of NVMPoints
    '''
    def __init__(self):
        '''Initializes NVMModel with empty cameras and empty points'''
        self.cameras = []
        self.points = []

    def from_file(self, nvm_file, image_path):
        '''Given file object, reads lines and creates NVM object'''
        # first get number of cameras
        num_camera_str = nvm_file.readline().strip()
        while(not is_int(num_camera_str)):
            num_camera_str = nvm_file.readline().strip()
        num_camera = int(num_camera_str)

        # next, parse cameras
        logging.info('[NVM Model Parsing] Parsing Camera')

        for camera_index in range(0, num_camera):
            camera_line = nvm_file.readline()
            self.cameras.append(self.parse_camera_line(camera_line, image_path))

        # parse number of points
        logging.info('[NVM Model Parsing] Parsing Points')
        num_points_str = nvm_file.readline().strip()
        while(not is_int(num_points_str)):
            num_points_str = nvm_file.readline().strip()

        num_points = int(num_points_str)
        for point_index in range(0, num_points):
            point_line = nvm_file.readline()
            self.points.append(self.parse_point_line(point_line))

        logging.info('[NVM Model Parsing] Parsing Done')

    def parse_camera_line(self, camera_line, image_path):
        '''Parses camera line from NVM camera line'''
        # filename focal_length quaternion(wxyz) position(xyz)
        tokens = camera_line.split()
        filename = tokens[0]

        # parse camera name
        name_token = filename.split('/')
        image_file_name = name_token[-1]
        camera_name = name_token[-1].split('.')[0]
        image_file_path = path.join(image_path, image_file_name)

        with Image.open(image_file_path) as img:
            width, height = img.size

        # parse camera position
        position_x = tokens[6]
        position_y = tokens[7]
        position_z = tokens[8]
        position_arr = np.array([
            float(position_x),
            float(position_y),
            float(position_z)
        ])
        position = Point()
        position.from_array(position_arr)

        # parse camera rotation
        rotation_w = tokens[2]
        rotation_x = tokens[3]
        rotation_y = tokens[4]
        rotation_z = tokens[5]

        rotation_quat = np.array([
            float(rotation_w),
            float(rotation_x),
            float(rotation_y),
            float(rotation_z)
        ])
        rotation = Rotation()
        rotation.from_quaternion(rotation_quat)
        rotation.from_matrix(np.transpose(rotation.to_matrix()))

        # parse focal length
        focal_length = float(tokens[1])

        # parse radial distortion (used in NVM)
        radial_distortion = float(tokens[9])

        # extra principal offset values
        cx = width/2
        cy = height/2
        camera = NVMCamera(camera_name,position,rotation,focal_length,
                           cx, cy, radial_distortion)
        return camera

    def parse_point_line(self, point_line):
        '''Parses point line from NVM point line

        point line line format:
        pos(xyz) color(rgb) Num_measurement [Measurement] 
        Measurement format:
        camera_index Feature_Index screen_pos(xy)
        '''
        tokens = point_line.split(' ')

        # parse camera position
        position_x = tokens[0]
        position_y = tokens[1]
        position_z = tokens[2]
        position_arr = np.array([
            float(position_x),
            float(position_y),
            float(position_z)
        ])
        position = Point().from_array(position_arr)
        color_x = tokens[3]
        color_y = tokens[4]
        color_z = tokens[5]
        color_arr = np.array([
            float(color_x),
            float(color_y),
            float(color_z)
        ])

        point = NVMPoint(position,color_arr)

        # parse measurement
        num_measurement = int(tokens[6])
        for i in range(0,num_measurement):
            parse_index = 7 + i*4
            camera_index = int(tokens[parse_index+0])
            feature_index = int(tokens[parse_index+1])
            screen_pos_x = tokens[parse_index+2]
            screen_pos_y = tokens[parse_index+3]
            camera = self.cameras[camera_index]
            point.add_measurement(camera, feature_index, screen_pos_x, screen_pos_y)

        return point

