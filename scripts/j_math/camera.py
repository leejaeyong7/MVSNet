import numpy as np

from j_math.rotation import Rotation
from j_math.point import Point

class Camera:
    def __init__(self, position, rotation, fx, fy, cx=0, cy=0, k1=0, k2=0, p1=0, p2=0, k3=0):
        '''
        extrinsic stores: [R | t]
        position should be C, rotation should be R_c
        
        R = R_c ^t
        t = -R C
        C = -R_c t
        '''
        self.extrinsic = np.eye(4, dtype=np.float)
        self.intrinsic = np.eye(3, dtype=np.float)
        self.radial_distortion = np.array([1, 0, 0, 0])
        self.tangential_distortion = np.array([0, 0])
        self.set_rotation(rotation)
        self.set_position(position)
        self.set_intrinsic(fx, fy, cx, cy)
        self.set_distortion(k1, k2, p1, p2, k3)

    def get_position(self):
        ''' returns camera center in world coordinates
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

    def get_rotation(self):
        ''' returns camera rotation matrix
        i.e) From R, return R_c
        '''
        rot = self.extrinsic[0:3, 0:3]
        return Rotation().from_matrix(np.transpose(rot))

    def set_position(self, position):
        ''' sets caemra translation by camera center
        Given C, save t
        NOTE)
        t = -RC
        '''
        rot  = self.extrinsic[0:3, 0:3]
        cc = position.to_array()
        trans = -np.matmul(rot, cc)
        self.extrinsic[0:3, 3] = trans

    def set_rotation(self, rotation):
        '''
        Given R_c, save R
        NOTE)
        R = Rc ^T
        '''
        self.extrinsic[0:3, 0:3] = np.transpose(rotation.to_matrix())

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

    def get_focal_length(self):
        return (self.intrinsic[0, 0] + self.intrinsic[1, 1]) / 2

    def homography_to(self, target, depth):
        ''' returns homography from self to target camera 
        i.e) pixel from self camera passing this homography matrix
             will return target camera's pixel coordinate
        '''
        target_r = target.extrinsic[0:3, 0:3]
        target_k = target.intrinsic
        self_r_inv = np.transpose(self.extrinsic[0:3, 0:3])
        self_k_inv = np.linalg.inv(self.intrinsic)
        self_trans = self.extrinsic[0:3, 3]
        target_trans = target.extrinsic[0:3, 3]
        cam_rot = self.get_rotation().to_matrix()
        look_at = -cam_rot.dot(np.array([0,0,1]))
        principal_axis = np.transpose(look_at)

        homography = np.eye(3) - ((self_trans - target_trans).dot(principal_axis)  / depth)

        return target_k.dot(target_r).dot(homography).dot(self_r_inv).dot(self_k_inv)

    def get_projection_matrix(self):
        ''' returns 3x4 projection matrix 
        '''
        return self.intrinsic.dot(self.extrinsic[0:3, :])

    def get_inverse_project_function(self):
        ''' Inverse projection returns a line equation that gives world coordinate given distance
        
        returns scale (3x3 matrix) and a constant (3x1 array)
        to get the world coordinate, one must:
        pixel = [u, v, 1]
        depth = z
        (depth * scale * pixel) + pos

        P = K^-1 * pixel
        depth * [R | t]^-1 *P = X
        '''
        inv_proj_matrix = np.linalg.inv(self.intrinsic)

        rot = self.get_rotation().to_matrix()
        pos = self.get_position().to_array()
        scale = np.matmul(rot, inv_proj_matrix)

        return lambda p, depth: depth*scale.dot(p) + pos

    def get_inverse_project_line_equation(self):
        ''' Inverse projection returns a line equation that gives world coordinate given distance
        
        returns scale (3x3 matrix) and a constant (3x1 array)
        to get the world coordinate, one must:
        pixel = [u, v, 1]
        depth = z
        (depth * scale * pixel) + pos

        P = K^-1 * pixel
        depth * [R | t]^-1 *P = X
        '''
        inv_proj_matrix = np.linalg.inv(self.intrinsic)

        rot = self.get_rotation().to_matrix()
        pos = self.get_position().to_array()
        scale = np.matmul(rot, inv_proj_matrix)

        return  scale, pos

    def get_inverse_homogeneous_projection(self):
        inv_proj_matrix = np.linalg.inv(self.intrinsic)
        inv_4x4_proj_matrix = np.eye(4)
        inv_4x4_proj_matrix[0:3, 0:3] = inv_proj_matrix
        inv_camera_matrix = np.linalg.inv(self.extrinsic)
        return inv_camera_matrix.dot(inv_4x4_proj_matrix)
        
    def get_homogeneous_projection(self):
        proj_matrix = np.linalg.inv(self.intrinsic)
        _4x4_proj_matrix = np.eye(4)
        _4x4_proj_matrix[0:3, 0:3] = proj_matrix
        return _4x4_proj_matrix.dot(self.extrinsic)
        
