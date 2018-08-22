import numpy as np

from j_math.rotation import Rotation
from j_math.point import Point

class Camera:
    def __init__(self, position, rotation, fx, fy, cx=0, cy=0, k1=0, k2=0, p1=0, p2=0, k3=0):
        self.extrinsic = np.eye(4, dtype=np.float)
        self.intrinsic = np.eye(3, dtype=np.float)
        self.radial_distortion = np.array([1, 0, 0, 0])
        self.tangential_distortion = np.array([0, 0])
        self.set_position(position)
        self.set_rotation(rotation)
        self.set_intrinsic(fx, fy, cx, cy)
        self.set_distortion(k1, k2, p1, p2, k3)

    def get_position(self):
        return Point().from_array(self.extrinsic[3, 0:3])

    def get_rotation(self):
        return Rotation().from_matrix(self.extrinsic[0:3, 0:3])

    def set_position(self, position):
        self.extrinsic[3, 0:3] = position.to_array()

    def set_rotation(self, rotation):
        self.extrinsic[0:3, 0:3] = rotation.to_matrix()

    def get_intrinsic(self):
        return self.intrinsic

    def set_intrinsic(self, fx, fy, cx, cy):
        self.intrinsic[0, 0] = fx
        self.intrinsic[1, 1] = fy
        self.intrinsic[0, 2] = cx
        self.intrinsic[1, 2] = cy

    def set_distortion(self, k1, k2, p1, p2, k3):
        self.radial_distortion[1:] = [k1, k2, k3]
        self.tangential_distortion[0:2] = [p1, p2]
