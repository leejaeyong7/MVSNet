import time
import numpy as np
import tensorflow as tf
from mvs.mvs_camera import MVSCamera
import re
import cv2
import logging
import os
import shutil
import matplotlib.pyplot as plt
logging.basicConfig(level='INFO')
''' reads PFM file (from MVSNet output) and writes to PLY file '''

def mkdirp(dest):
    try: 
        os.makedirs(dest)
    except OSError:
        if not os.path.isdir(dest):
            raise
PROJECT='nbmtech'
# MVS_OUTPUT_PATH = '/Users/jae/Research/dataset/outputs/mvsnet/{}/depths_mvsnet'.format(PROJECT)
# OUTPUT_FOLDER = '/Users/jae/Research/outputs/{}'.format(PROJECT)

MVS_OUTPUT_PATH = '/home/ubuntu/output/{}/depths_mvsnet'.format(PROJECT)
OUTPUT_FOLDER = '/home/ubuntu/pointcloud/{}_test'.format(PROJECT)

mkdirp(OUTPUT_FOLDER)


def verify_images(ref_proj_mat, ref_inv_proj_func,
                  source_proj_mat, source_inv_proj_func,
                  ref_depth_map, source_depth_map, w, h):
    (im_h, im_w) = ref_depth_map.shape
    time_start = time.time()

    # obtain source pixel from w / h / ref_depth_map
    ref_pixel = np.array([w, h])
    ref_depth = ref_depth_map[h, w]
    if(ref_depth <= 0):
        return False
    ref_homo_pixel = np.array([w, h, 1])

    ref_point = ref_inv_proj_func(ref_homo_pixel, ref_depth)
    source_reproj_homo_pixel = source_proj_mat.dot(np.append(ref_point, [1]))

    source_pixel = source_reproj_homo_pixel / source_reproj_homo_pixel[2]
    source_pixel_rounded = source_pixel[0:2].astype(int)

    # check if pixel_rounded is inside source image
    if(source_pixel_rounded[0] < 0 or source_pixel_rounded[0] >= im_w or
       source_pixel_rounded[1] < 0 or source_pixel_rounded[1] >= im_h):
        # if not, return false
        return False

    # obtain depth value of projected pixel in source image
    source_depth = source_depth_map[source_pixel_rounded[1],
                                    source_pixel_rounded[0]]
    if(source_depth <= 0):
        return False
    # source_to_ref_homography = source_camera.homography_to(ref_camera, source_depth)

    source_point = source_inv_proj_func(source_pixel, source_depth)

    ref_reproj_homo_pixel = ref_proj_mat.dot(np.append(source_point, [1]))
    ref_reproj_pixel = ref_reproj_homo_pixel / ref_reproj_homo_pixel[2]
    ref_reproj_pixel_rounded = ref_reproj_pixel[0:2].astype(int)

    # check if reprojected point is inside image boundary
    if(ref_reproj_pixel_rounded[0] < 0 or ref_reproj_pixel_rounded[0] >= im_w or
       ref_reproj_pixel_rounded[1] < 0 or ref_reproj_pixel_rounded[1] >= im_h or
       source_depth <= 0):
        # if not, return false
        return False
    ref_reproj_depth = ref_depth_map[ref_reproj_pixel_rounded[1],
                                     ref_reproj_pixel_rounded[0]]

    # obtain reprojected_pixel from w / h / ref_camera
    pixel_diff = np.sum(np.abs(ref_homo_pixel - ref_reproj_pixel))
    depth_diff = abs(ref_depth - ref_reproj_depth) / ref_depth

    return pixel_diff < 1 and depth_diff < 0.01


def post_process(cameras, depth_maps, prob_maps):
    ''' post processes depth maps by photometric / geometric verification
    
    Photometric:
      - All depth maps below 0.8 probability should be removed
    Geometric:
      - All depths should satisfy at least 3 views that meets following
        formula:
        | p_reproj - p_1 | < 1
        | d_reproj - d_1 | / d_1  < 0.01
        p_reproj = reprojected pixel that was projected by reprojected pixel of other view
        d_reproj = reprojected depth that was projected by reprojected depth of other view
    INPUT:
        N mvs_cameras
        NxWxH depth maps
        NxWxH prob maps
    OUTPUT:
        NxWxH depth maps
    '''

    # perform photometric verification
    logging.info('[POST PROCESS] verifying photometric verification...')
    photometric_verified = np.where(prob_maps < 0.8, 0, depth_maps)
    return photometric_verified
    logging.info('[POST PROCESS] photometric verification verified!')

    # perform geoemtric verification
    (N, H, W) = prob_maps.shape
    # visited = np.zeros((N, H, W), dtype=bool)
    verified = np.zeros((N, H, W), dtype=np.int32)

    logging.info('[POST PROCESS] verfiying geometric verification...')
    # for each image's W / H, compare other image
    # 1. for all depth map images, get world coordinate values
    all_world_points = np.zeros((4, N*W*H))
    logging.info('[POST PROCESS] loading per camera points')
    for i, camera in enumerate(cameras):
        # 4x4 inv homo proj
        inv_proj = camera.get_inverse_homogeneous_projection()
        i_xs, i_ys = np.meshgrid(range(W), range(H))

        # (0,0), (1,0) ... (W-1, 0), (0,1) ... (W-1,H-1)
        image_coords = np.array([i_xs.flatten(), i_ys.flatten()]).T 
        ones = np.ones((W*H, 1))
        depth_map = depth_maps[i, :, :]
        inv_depths = np.reshape(1 / depth_map, (W*H, 1))
        h_image_coords = np.hstack((image_coords, ones, inv_depths))
        world_points = inv_proj.dot(h_image_coords.T)
        all_world_points[:, i*W*H:(i+1)*W*H] = world_points

    # project 3d points back into camera
    for i in range(0, N-1):
        depth_map = depth_maps[i, :, :]
        depths = depth_map.flatten()
        logging.info('[POST PROCESS] matching camera pairs {}/{}'.format(i+1, len(cameras)))
        for j in range(i+1, N):
            #start = time.time()
            ref_camera = cameras[i]
            source_camera = cameras[j]
            ref_proj = ref_camera.get_projection_matrix()
            source_proj = source_camera.get_projection_matrix()
            ref_points = all_world_points[:, i*W*H:(i+1)*W*H]
            source_points = all_world_points[:, j*W*H:(j+1)*W*H]

            ref_reproj_homo = ref_proj.dot(source_points)
            ref_reproj = ref_reproj_homo / ref_reproj_homo[2,:]

            source_reproj_homo = source_proj.dot(ref_points)
            source_reproj = source_reproj_homo / source_reproj_homo[2,:]
            source_coords = np.round(source_reproj).astype(int)
            source_indices = source_coords[0,:] + source_coords[1,:]*W
            source_valids_w = np.logical_and(source_coords[0,:] >= 0, source_coords[0,:] < W)
            source_valids_h = np.logical_and(source_coords[1,:] >= 0, source_coords[1,:] < H)
            source_valids = np.logical_and(source_valids_w, source_valids_h)

            # go through each W / H in ref, 
            all_ref_indices = np.array(range(W*H))

            # ref / source indices that has valid source reprojection
            source_valid_ref_indices = all_ref_indices[source_valids] 
            source_valid_source_indices = source_indices[source_valids]

            # values valid from source projection
            # this value contains pixel values of points in reference camera, in order of all_source_valid_ref_indices
            source_valid_ref_reproj = ref_reproj[:, source_valid_source_indices]
            source_valid_ref_reproj_coords = np.round(source_valid_ref_reproj).astype(int)
            source_valid_ref_reproj_indices = source_valid_ref_reproj_coords[0,:] + source_valid_ref_reproj_coords[1,:]*W
            source_valid_ref_valids_w = np.logical_and(source_valid_ref_reproj_coords[0,:] >= 0, source_valid_ref_reproj_coords[0,:] < W)
            source_valid_ref_valids_h = np.logical_and(source_valid_ref_reproj_coords[1,:] >= 0, source_valid_ref_reproj_coords[1,:] < H)
            source_valid_ref_valids = np.logical_and(source_valid_ref_valids_w, source_valid_ref_valids_h)

            # values that are both valid in reprojection of source projected points
            source_ref_valid_ref_reproj_indices = source_valid_ref_reproj_indices[source_valid_ref_valids]
            source_ref_valid_ref_indices = source_valid_ref_indices[source_valid_ref_valids]
            source_ref_valid_ref_depths = depths[source_ref_valid_ref_reproj_indices]
            source_ref_valid_source_indices = source_valid_source_indices[source_valid_ref_valids]

            original_depths = depths[source_ref_valid_ref_indices]
            depth_diffs = np.abs(original_depths - source_ref_valid_ref_depths) / original_depths

            original_pixel_x, original_pixel_y = np.divmod(source_ref_valid_ref_indices, W)
            reproj_pixels = ref_reproj[:, source_ref_valid_source_indices]

            pixel_diffs = np.abs((original_pixel_x - reproj_pixels[0,:]) + (original_pixel_y - reproj_pixels[1,:]))

            pixel_is_valid = pixel_diffs < 1
            depth_is_valid = depth_diffs < 0.01

            is_valid = np.logical_and(pixel_is_valid, depth_is_valid)

            valid_ref_indices = source_ref_valid_ref_indices[is_valid]
            valid_source_indices = source_ref_valid_source_indices[is_valid]

            verified_i = np.zeros((W*H, 1), dtype=int)
            verified_i[valid_ref_indices] = 1
            verified_j = np.zeros((W*H, 1), dtype=int)
            verified_j[valid_source_indices] = 1
            verified[i, :, :] += np.reshape(verified_i, (H, W))
            verified[j, :, :] += np.reshape(verified_j, (H, W))
            #end = time.time()
            #print(end - start)

    verified_depth_map = np.where(verified >= 3, photometric_verified, 0)
    logging.info('[POST PROCESS] geometric verification verified!')

    # return verified depth maps
    return verified_depth_map


def load_pfm(file_path):
    with open(file_path, 'r') as file:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = str(file.readline()).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        # data = np.fromfile(file, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data

def load_files (file_path):
    logging.info('[LOAD FILES] parsing depth maps')
    depth_maps = np.array([
        load_pfm(os.path.join(file_path,filename))
        for filename in sorted(os.listdir(file_path))
        if filename.endswith('.pfm') and '_' not in filename
    ])

    logging.info('[LOAD FILES] parsing prob maps')
    prob_maps = np.array([
        load_pfm(os.path.join(file_path,filename))
        for filename in sorted(os.listdir(file_path))
        if filename.endswith('prob.pfm')
    ])
    logging.info('[LOAD FILES] parsing cameras')
    cameras = [ 
        MVSCamera.from_file(os.path.join(file_path,filename))
        for filename in sorted(os.listdir(file_path))
        if filename.endswith('.txt')
    ]
    logging.info('[LOAD FILES] parsing image files')
    image_files = [ 
        os.path.join(file_path, filename)
        for filename in sorted(os.listdir(file_path))
        if filename.endswith('.png')
    ]
    return image_files, cameras, depth_maps, prob_maps

def merge_depth_maps(cameras, image_files, depth_maps):
    ''' merges image + depth into sindle PLY file '''
    global_point_cloud = []
    for image_id, image_file in enumerate(image_files):
        logging.info('image_id {} / {}'.format(image_id, len(image_files)))
        image = cv2.imread(image_file)
        image = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 
        (H, W, dim) = image.shape
        camera = cameras[image_id]
        camera_inv_proj_line , camera_center = camera.get_inverse_project_line_equation()
        for width in range(W):
            for height in range(H):
                depth = depth_maps[image_id, height, width]
                if(depth <= 0):
                    continue
                pixel_coord = np.array([width, height, 1])
                world_pos = depth * np.matmul(camera_inv_proj_line, pixel_coord) + camera_center
                color = image[height, width, :]
                pos_list = world_pos.tolist()
                color_list = list(reversed(color.tolist()))

                global_point_cloud.append(pos_list + color_list)

    ply_file_path = os.path.join(OUTPUT_FOLDER, 'final.ply')
    with open(ply_file_path, 'w') as ply_file:
        header = [
            'ply',
            'format ascii 1.0',
            'element vertex {}'.format(len(global_point_cloud)),
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header',
            '',
        ]
        ply_file.write('\n'.join(header))
        for point in global_point_cloud:
            point_data = ' '.join([str(d) for d in point])
            ply_file.write('{}\n'.format(point_data))

def visibility_based_fusion(cameras, image_files, depth_maps):
    return

def main():
    image_files, cameras, depth_maps, prob_maps = load_files(MVS_OUTPUT_PATH)
    verified_depth_maps = post_process(cameras, depth_maps, prob_maps)
    for image_id, image_file in enumerate(image_files):
        depth_map = verified_depth_maps[image_id, :, :]
        plt.imsave('{}/{:08d}_depth.png'.format(OUTPUT_FOLDER, image_id), depth_map, cmap='rainbow')
    merge_depth_maps(cameras, image_files, verified_depth_maps)
    '''
    cameras = cameras[0:500]
    depth_maps = depth_maps[0:500,:,:]
    prob_maps = prob_maps[0:500,:,:]
    image_files = image_files[0:500]
        
    verified_depth_maps = post_process(cameras, depth_maps, prob_maps)
    for image_id, image_file in enumerate(image_files):
        depth_map = verified_depth_maps[image_id, :, :]
        plt.imsave('{}/{:08d}_depth.png'.format(OUTPUT_FOLDER, image_id), depth_map, cmap='rainbow')
    merge_depth_maps(cameras, image_files, verified_depth_maps)
    '''
main()

# depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_HOT)
