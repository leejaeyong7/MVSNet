import numpy as np

class Rotation:
    def __init__(self, ):
        self.matrix = np.eye(3)

    def from_quaternion(self, q):
        ''' quaternion to rotation matrix
        Uses following formula:
        1 - 2*qy2 - 2*qz2	2*qx*qy - 2*qz*qw	2*qx*qz + 2*qy*qw
        2*qx*qy + 2*qz*qw	1 - 2*qx2 - 2*qz2	2*qy*qz - 2*qx*qw
        2*qx*qz - 2*qy*qw	2*qy*qz + 2*qx*qw	1 - 2*qx2 - 2*qy2
        '''
        n = numpy.dot(q, q)
        _EPS = numpy.finfo(float).eps * 4.0
        if n < _EPS:
            return numpy.identity(4)
        q *= math.sqrt(2.0 / n)
        q = numpy.outer(q, q)
        self.matrix = numpy.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
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

