from os import path

import cv2
import numpy as np

from j_math.camera import Camera

class MVSCamera(Camera):
    def __init__(self, index, position, rotation, fx, fy, cx, cy, depth_min, depth_interval):
        Camera.__init__(self, position, rotation, fx, fy, cx, cy)
        self.index = index
        self.depth_min = depth_min
        self.depth_interval = depth_interval
        self.image_data = None

    def read_image_file(self, file_path):
        self.image_data = cv2.imread(file_path)

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
            '{} {}'.format(self.depth_min, self.depth_interval)
        ])
        

    def to_file(self, output_path):
        image_file_name = '{:08d}.jpg'.format(self.index)
        image_file_path = path.join(output_path, 'images', image_file_name)
        # write image iff exists
        if(not self.image_data):
            cv2.imwrite(image_file_path, self.image_data)
        else:
            raise Exception('No image found')

        camera_file_name = '{:08d}_cam.txt'.format(self.index)
        camera_file_path = path.join(output_path, 'cams', camera_file_name)

        # write camera
        with open(camera_file_path, 'w') as f:
            f.writelines(self.to_string())
