import numpy as np
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
PROJECT='gilbane'
MVS_OUTPUT_PATH = '/Users/jae/Research/dataset/outputs/mvsnet/{}/depths_mvsnet'.format(PROJECT)
OUTPUT_FOLDER = '/Users/jae/Research/outputs/{}'.format(PROJECT)
mkdirp(OUTPUT_FOLDER)


def verify_images(ref_proj_mat, ref_inv_proj_func,
                  source_proj_mat, source_inv_proj_func,
                  ref_depth_map, source_depth_map, w, h):
    (im_h, im_w) = ref_depth_map.shape
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
    logging.info('[POST PROCESS] photometric verification verified!')

    # perform geoemtric verification
    (N, H, W) = prob_maps.shape
    # visited = np.zeros((N, H, W), dtype=bool)
    verified = np.zeros((N, H, W), dtype=np.int32)

    logging.info('[POST PROCESS] verfiying geometric verification...')
    # for each image's W / H, compare other image
    for ref_image_id in range(0, N):
        logging.info('[POST PROCESS] verifying geometric verification ({}/{})'.format(ref_image_id+1, N))
        for source_image_id in range(ref_image_id , N):
            if(ref_image_id == source_image_id):
                continue
            ref_camera = cameras[ref_image_id]
            source_camera = cameras[source_image_id]
            ref_depth_map = photometric_verified[ref_image_id, :, :]
            source_depth_map = photometric_verified[source_image_id, :, :]
            ref_projection_mat = ref_camera.get_projection_matrix()
            source_projection_mat = source_camera.get_projection_matrix()
            ref_inv_projection_func= ref_camera.get_inverse_project_function()
            source_inv_projection_func= source_camera.get_inverse_project_function()
            for width in range(0, W):
                for height in range(0, H):
                    if(verify_images(ref_projection_mat, ref_inv_projection_func,
                                     source_projection_mat, source_inv_projection_func,
                                     ref_depth_map, source_depth_map,
                                     width, height)):
                        verified[ref_image_id, height, width] += 1
                        verified[source_image_id, height, width] += 1

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
        
main()

# depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_HOT)
