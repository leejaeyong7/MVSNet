from nvm.nvm import NVM
import numpy as np
from mvs.mvs_camera import MVSCamera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os

file_dir = '../../dataset/outputs/mvsnet/dtu/depths_mvsnet'
nvm_file_dir = '../../dataset/inputs/mvsnet/dtu/reconstruction0.nvm'
nvm_image_dir = '../../dataset/inputs/mvsnet/dtu/images'

file_list = os.listdir(file_dir)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def create_frus(arr, o, q1, q2, q3, q4, q5, index):
    # o => 1 => 2 => o
    arr.append(o[index])
    arr.append(q1[index])
    arr.append(q2[index])
    arr.append(o[index])

    # o => 2 => 3 => o
    arr.append(o[index])
    arr.append(q2[index])
    arr.append(q3[index])
    arr.append(o[index])

    # o => 3 => 4 => o
    arr.append(o[index])
    arr.append(q3[index])
    arr.append(q4[index])
    arr.append(o[index])

    # o => 4 => 1 => o
    arr.append(o[index])
    arr.append(q4[index])
    arr.append(q1[index])
    arr.append(o[index])

    arr.append(o[index])
    arr.append(q5[index])
    arr.append(o[index])

# reading from file list
for file_path in sorted(file_list):
    if(file_path == '.' or file_path == '..'  or file_path == '.DS_Store'):
        continue

    if(not file_path.endswith('.txt')):
        continue

    full_file_path = os.path.join(file_dir, file_path)

    camera = MVSCamera.from_file(full_file_path)

    fl = camera.get_focal_length()
    xs = []
    ys = []
    zs = []

    cam_pos = camera.get_position().to_array()
    cam_rot = camera.get_rotation().to_matrix()
    look_at = np.matmul(cam_rot, np.array([0,0,1]))
    w = camera.intrinsic[0, 2] / fl
    h = camera.intrinsic[1, 2] / fl
    f = camera.intrinsic[0, 0] / fl

    quat_1 = np.matmul(cam_rot, np.array([w,h,f]))
    quat_2 = np.matmul(cam_rot, np.array([w,-h,f]))
    quat_3 = np.matmul(cam_rot, np.array([-w,-h,f]))
    quat_4 = np.matmul(cam_rot, np.array([-w,h,f]))
    quat_5 = np.matmul(cam_rot, np.array([0,h,0]))

    scale = 55

    cam_look_at = np.add(cam_pos, look_at*scale)
    cam_quat1  = np.add(cam_pos, quat_1*scale)
    cam_quat2  = np.add(cam_pos, quat_2*scale)
    cam_quat3  = np.add(cam_pos, quat_3*scale)
    cam_quat4  = np.add(cam_pos, quat_4*scale)
    cam_quat5  = np.add(cam_pos, quat_5*scale)

    create_frus(xs, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 0)
    create_frus(ys, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 1)
    create_frus(zs, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 2)
    ax.plot(xs, ys, zs=zs)
plt.axis('equal')
plt.show()


# # reading from nvm
# nvm_object= NVM().from_file(nvm_file_dir, nvm_image_dir)
# nvm_model = nvm_object.models[0]

# for camera in nvm_model.cameras:
#     fl = camera.get_focal_length()
#     xs = []
#     ys = []
#     zs = []

#     cam_pos = camera.get_position().to_array()
#     cam_rot = camera.get_rotation().to_matrix()
#     look_at = np.matmul(cam_rot, np.array([0,0,1]))
#     w = camera.intrinsic[0, 2] / fl
#     h = camera.intrinsic[1, 2] / fl
#     f = camera.intrinsic[0, 0] / fl

#     quat_1 = np.matmul(cam_rot, np.array([w,h,f]))
#     quat_2 = np.matmul(cam_rot, np.array([w,-h,f]))
#     quat_3 = np.matmul(cam_rot, np.array([-w,-h,f]))
#     quat_4 = np.matmul(cam_rot, np.array([-w,h,f]))
#     quat_5 = np.matmul(cam_rot, np.array([0,h,0]))

#     scale = 5

#     cam_look_at = np.add(cam_pos, look_at*scale)
#     cam_quat1  = np.add(cam_pos, quat_1*scale)
#     cam_quat2  = np.add(cam_pos, quat_2*scale)
#     cam_quat3  = np.add(cam_pos, quat_3*scale)
#     cam_quat4  = np.add(cam_pos, quat_4*scale)
#     cam_quat5  = np.add(cam_pos, quat_5*scale)

#     create_frus(xs, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 0)
#     create_frus(ys, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 1)
#     create_frus(zs, cam_pos, cam_quat1, cam_quat2, cam_quat3, cam_quat4, cam_quat5, 2)
#     ax.plot(xs, ys, zs=zs)

# px = []
# py = []
# pz = []
# for point in nvm_model.points:
#     px.append(point.position[0])
#     py.append(point.position[1])
#     pz.append(point.position[2])
# ax.scatter(px, py, zs=pz)
# plt.axis('equal')
# plt.show()
