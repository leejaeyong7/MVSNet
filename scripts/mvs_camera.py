import numpy as np
from os import path

class MVSCamera:
    def __init__(self, index, rotation=None, position=None,
                 intrinsic=None, depth_min=None, depth_interval=None):
        self.index = 0
        if(extrinsic == None):
            self.extrinsic = np.eye(4, dtype=np.float)
        else:
            self.extrinsic = extrinsic
        if(intrinsic == None):
            self.intrinsic = np.eye(3, dtype=np.float)
        else:
            self.intrinsic = intrinsic

        self.depth_min = 0
        self.depth_interval = 1
        self.depth_max = DEPTH_MIN + (interval_scale * DEPTH_INTERVAL) * (max_d - 1)

    def to_string(self):
        return '\n'.join([
            'extrinsic',
            ' '.join(str(i) for i in self.extrinsic[0]),
            ' '.join(str(i) for i in self.extrinsic[1]),
            ' '.join(str(i) for i in self.extrinsic[2]),
            ' '.join(str(i) for i in self.extrinsic[3]),
            '',
            'intrinsic',
            ' '.join(str(i) for i in self.intrinsic[0]),
            ' '.join(str(i) for i in self.intrinsic[1]),
            ' '.join(str(i) for i in self.intrinsic[2]),
            '',
            '{} {}'.format(self.depth_min, self.depth_interval)
        ])
        

    def to_file(self, output_path):
        file_name = '{:08d}_cam.txt'.format(self.index)
        file_path = path.join(output_path, file_name)
        with open(file_path, 'w') as f:
            f.writelines(self.to_string())
