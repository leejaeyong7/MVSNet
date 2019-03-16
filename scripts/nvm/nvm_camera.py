from j_math.camera import Camera
from j_math.point import Point

class NVMCamera(Camera):
    '''Wrapper for NVM Camera that inherits generic camera model

    NOTE) NVM file's camera line is formatted as:
    <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
    '''
    def __init__(self, name, position, rotation, fl, cx, cy, radial_distortion):
        r = radial_distortion
        # TODO: implement radial distortion for NVM camera
        Camera.__init__(self, position, rotation, fl, fl, cx, cy)
        self.name = name
        self.features = {}

    def attach_point(self, point, index):
        self.features[index] = point

