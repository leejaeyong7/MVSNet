class MVSImage:
   def __init__(self, index):
      self.index = 
      self.data = None

   def from_file(self, file_path):
      self.data = cv2.imread(file_path)
   
