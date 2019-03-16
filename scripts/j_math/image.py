import cv2
import numpy as np

class Image:
    def __init__(self, name, data=None):
        self.name = name
        self.data = None

    def from_file(self, file_path):
        self.data = cv2.imread(file_path)
    
    def get_name(self):
        return self.name
