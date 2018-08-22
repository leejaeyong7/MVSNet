import numpy as np

class Rotation:
    def __init__(self, ):
        self.matrix = np.eye(3)

    def from_quaternion(self, quaternion):
        ''' quaternion to rotation matrix
        Uses following formula:
        1 - 2*qy2 - 2*qz2	2*qx*qy - 2*qz*qw	2*qx*qz + 2*qy*qw
        2*qx*qy + 2*qz*qw	1 - 2*qx2 - 2*qz2	2*qy*qz - 2*qx*qw
        2*qx*qz - 2*qy*qw	2*qy*qz + 2*qx*qw	1 - 2*qx2 - 2*qy2
        '''
        w = quaternion[0]
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]
        self.matrix[0, 0] = 1 - 2*y*y - 2*z*z
        self.matrix[0, 1] = 2*x*y - 2*z*w
        self.matrix[0, 2] = 2*x*z - 2*y*w

        self.matrix[1, 0] = 2*x*y - 2*z*w
        self.matrix[1, 1] = 1 - 2*x*x - 2*z*z
        self.matrix[1, 2] = 2*y*z - 2*x*w

        self.matrix[2, 0] = 2*x*z - 2*y*w
        self.matrix[2, 1] = 2*y*z - 2*x*w
        self.matrix[2, 2] = 1 - 2*x*x - 2*y*y
        return self

    def from_eulers(self, x, y, z):
        return self

    def from_matrix(self, matrix):
        self.matrix = matrix.copy()
        return self

    def to_quaternion(self):
        pass

    def to_eulers(self):
        pass

    def to_matrix(self):
        return self.matrix

