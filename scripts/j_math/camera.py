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
        ''' returns camera center
        NOTE)
        R = Rc ^T
        t = -RC
        C = -Rc t
        return C
        '''
        rot_c  = self.get_rotation().to_matrix()
        trans = self.extrinsic[0:3, 3]
        cc = -np.matmul(rot_c, trans)
        return Point().from_array(cc)
        # return Point().from_array(trans)

    def get_rotation(self):
        return Rotation().from_matrix(np.transpose(self.extrinsic[0:3, 0:3]))

    def set_position(self, position):
        ''' sets camera position from camera center 
        NOTE)
        R = Rc ^T
        t = -RC
        return t
        '''
        rot  = self.extrinsic[0:3, 0:3]
        cc = position.to_array()
        trans = -np.matmul(rot, cc)
        self.extrinsic[0:3, 3] = trans

    def set_rotation(self, rotation):
        # on updating rotation, we need to update translation too
        pos = self.get_position()
        self.extrinsic[0:3, 0:3] = np.transpose(rotation.to_matrix())
        self.set_position(pos)

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
