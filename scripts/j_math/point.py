import numpy as np

class Point:
    def __init__(self):
        self.position = np.array([0,0,0])

    def from_array(self, array):
        self.position = array.copy()
        return self

    def to_array(self):
        return self.position
    

    def distance_to(self, point):
        return np.linalg.norm(self.to_array() - point.to_array())
