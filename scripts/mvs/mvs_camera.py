from os import path
from shutil import copyfile

import cv2
import numpy as np

from j_math.camera import Camera
from j_math.point import Point
from j_math.rotation import Rotation

class MVSCamera(Camera):
    def __init__(self, index, position, rotation, fx, fy, cx, cy, depth_min, depth_interval):
        Camera.__init__(self, position, rotation, fx, fy, cx, cy)
        self.index = index
        self.depth_min = depth_min
        self.depth_interval = depth_interval
        self.image_path = None

    def set_image_file(self, file_path):
        self.image_path = file_path

    def to_string(self):
        return '\n'.join([
            'extrinsic',
            ' '.join(str(i) for i in self.extrinsic[0]),
            ' '.join(str(i) for i in self.extrinsic[1]),
            ' '.join(str(i) for i in self.extrinsic[2]),
            ' '.join(str(i) for i in self.extrinsic[3]),
            '',
            'intrinsic',
            ' '.join(str(i) for i in self.intrinsic[0]),
            ' '.join(str(i) for i in self.intrinsic[1]),
            ' '.join(str(i) for i in self.intrinsic[2]),
            '',
            '{} {}'.format(self.depth_min, self.depth_interval),
            ''
        ])

    def from_string(self, string):
        lines = string.splitlines()
        if lines[0] != 'extrinsic' or lines[6] != 'intrinsic':
            raise Exception('invalid format')
        extrinsic_r_0 = np.array([float(v) for v in lines[1].strip().split()])
        extrinsic_r_1 = np.array([float(v) for v in lines[2].strip().split()])
        extrinsic_r_2 = np.array([float(v) for v in lines[3].strip().split()])
        extrinsic_r_3 = np.array([float(v) for v in lines[4].strip().split()])
        extrinsic = np.array([extrinsic_r_0,
                               extrinsic_r_1,
                               extrinsic_r_2,
                               extrinsic_r_3])
        intrinsic_r_0 = np.array([float(v) for v in lines[7].strip().split()])
        intrinsic_r_1 = np.array([float(v) for v in lines[8].strip().split()])
        intrinsic_r_2 = np.array([float(v) for v in lines[9].strip().split()])
        intrinsic = np.array([intrinsic_r_0,
                               intrinsic_r_1,
                               intrinsic_r_2])

        depths = np.array([float(v) for v in lines[11].strip().split()])
        depth_min = depths[0]
        depth_int = depths[1]
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_interval = depth_int
        

    def to_file(self, output_path):
        image_file_name = '{:08d}.jpg'.format(self.index)
        image_file_path = path.join(output_path, 'images', image_file_name)
        # write image iff path exists
        if(self.image_path):
            copyfile(self.image_path, image_file_path)
        else:
            raise Exception('No image found')

        camera_file_name = '{:08d}_cam.txt'.format(self.index)
        camera_file_path = path.join(output_path, 'cams', camera_file_name)

        # write camera
        with open(camera_file_path, 'w') as f:
            f.writelines(self.to_string())

    @staticmethod
    def from_file(file_path):
        image_name = file_path.split('/')[-1]
        image_prefix = image_name[0:8]
        mvs_path = '/'.join(file_path.split('/')[0:-2])
        image_path = path.join(mvs_path, 'images', image_prefix + '.jpg')

        pos = Point()
        rot = Rotation()
        fx = 0
        fy = 0
        cx = 0
        cy = 0
        dm = 0
        di = 0

        mvs_camera = MVSCamera(0, pos, rot, fx, fy, cx, cy, dm, di)
        with open(file_path, 'r') as f:
            mvs_camera.from_string(f.read())
        mvs_camera.index = int(image_prefix)
        return mvs_camera
