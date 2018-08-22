from nvm_measurement import NVMMeasurement
from j_math.point import Point

class NVMPoint(Point):
    '''Generic object that describes NVM point
    
    initialized with position in 3D space, and color, contains 'add_measurement'
    function that adds current point to camera that captured it, with ID and 
    screen position
    '''
    def __init__(self, position, color):
        Point.__init__(self)
        # initialize as point
        self.from_array(position.to_array())
        # add extra fields used in NVM only
        self.color = color.copy()
        self.measurements = []
        self.cameras = {}

    def add_measurement(self, camera, feature_id, x, y):
        '''Adds measurement to Camera
        
        Adds measurement (feature) to camera, and adds camera's name in
        cameras hash for easy lookup.
        This enables refering to all cameras that tracked current point
        '''
        # add measurment
        self.measurements.append(NVMMeasurement(camera, feature_id, x, y))

        # attach point
        camera.attach_point(self, feature_id)

        # add to hash
        self.cameras[camera.name] = camera

