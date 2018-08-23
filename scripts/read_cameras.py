from mvs.mvs_camera import MVSCamera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

file_dir = '../../output/dtu/cams'
indices = [43, 44]
file_list = os.listdir(file_dir)
xs = []
ys = []
zs = []
for file_path in sorted(file_list):
    if(file_path == '.' or file_path == '..'  or file_path == '.DS_Store'):
        continue
    full_file_path = os.path.join(file_dir, file_path)
    camera = MVSCamera.from_file(full_file_path)
    # if(camera.index in indices):
    if(camera.index < 30):
        cam_pos = camera.get_position().to_array()
        xs.append(cam_pos[0])
        ys.append(cam_pos[1])
        zs.append(cam_pos[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs=zs)
plt.show()


        # print(file_path)
        # print(camera.get_position().to_array())
