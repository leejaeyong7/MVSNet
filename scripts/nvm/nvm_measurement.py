import numpy as np
class NVMMeasurement:
    '''Container for measurement value used in NVMPoint'''
    def __init__(self, camera, feature_id, x, y):
        '''initialize with image that captured this'''
        self.camera = camera
        self.feature_id = feature_id
        self.position = np.array([x,y])
    
