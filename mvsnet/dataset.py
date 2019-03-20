import random
import os
from os import path
import numpy as np
from data_formater import *


class MVSDataset():
    """
    TNT Data folder structure
    train/
        set_name
            depths/
            normals/
            cameras/
            images/
            pair.txt
    test/
        set_name
            cameras/
            images/
            pair.txt
    """
    def __init__(self, dataset_dir, num_neighbors, interval_scale, depth_interval, image_width, image_height, mode, target_set):
        self.dataset_dir = dataset_dir
        self.permute_neighbor = True
        self.depth_interval = depth_interval
        self.image_width = image_width
        self.image_height = image_height
        self.num_neighbors = num_neighbors

        self.mode = mode
        self.sample_list = []

        if(not target_set):
            target_sets = os.listdir(dataset_dir)
        else:
            target_sets = [target_set]
        # sample list should be array of path object where
        # paths = {
        #   images: [path],
        #   cameras: [path],
        #   depth: path, // for train mode
        #   normal: path, // for train mode
        # }
        for target_set in target_sets:
            project_dir = path.join(dataset_dir, target_set)
            image_dir = path.join(project_dir, 'images')
            camera_dir = path.join(project_dir, 'cameras')
            depth_dir = path.join(project_dir, 'depths')
            cluster_file = path.join(project_dir, 'pair.txt')

            # first fetch image list
            image_list = os.listdir(image_dir)
            # since we store image with sequential ids, image ids are simple
            # list of range
            image_ids = list(range(len(image_list)))
            if(mode == 'train' or mode =='eval'):
                with open(cluster_file, 'r') as cluster_file:
                    cluster_list = [
                        [int(w) for w in line.split()]
                        for line in cluster_file.readlines()
                    ]

            for image_id in image_ids:
                paths = {
                    'images': [],
                    'cameras': [],
                    'depth': None,
                }

                # check if mode is training
                # if so, we want to add images from cluster
                # else, we want to add ref image only
                ref_image_name = '%06d.png' % image_id
                ref_camera_name = '%06d_cam.txt' % image_id

                ref_image_path = path.join(image_dir, ref_image_name)
                ref_camera_path = path.join(camera_dir, ref_camera_name)

                paths['images'].append(ref_image_path)
                paths['cameras'].append(ref_camera_path)

                ref_depth_name = '%06d_cam.npz' % image_id
                ref_depth_path = path.join(depth_dir, ref_depth_name)
                if(mode == 'train'):
                    # choose 10 best neighbors
                    paths['depth'] = ref_depth_path
                    neighbors = cluster_list[image_id][:10]
                    for n in neighbors:
                        src_image_name = '%06d.png' % n
                        src_camera_name = '%06d_cam.txt' % n
                        src_image_path = path.join(image_dir, src_image_name)
                        src_camera_path = path.join(camera_dir,
                                                    src_camera_name)
                        paths['images'].append(src_image_path)
                        paths['cameras'].append(src_camera_path)
                elif(mode == 'eval'):
                    paths['depth'] = ref_depth_path
                    n = cluster_list[image_id][:10][0]
                    src_image_name = '%06d.png' % n
                    src_camera_name = '%06d_cam.txt' % n
                    src_image_path = path.join(image_dir, src_image_name)
                    src_camera_path = path.join(camera_dir,
                                                src_camera_name)
                    paths['images'].append(src_image_path)
                    paths['cameras'].append(src_camera_path)
                self.sample_list.append(paths)

    def __len__(self):
        return len(self.sample_list)

    def __iter__(self):
        while True:
            for index in range(len(self.sample_list)):
                paths = self.sample_list[index]
                image_paths = paths['images']
                camera_paths = paths['cameras']

                # randomly choose one of the neighbors
                if(self.mode == 'train'):
                    new_image_paths = []
                    new_camera_paths = []
                    num_neighbors = len(image_paths) - 1
                    n_indices = random.sample(range(num_neighbors),
                                              self.num_neighbors)
                    for n_index in n_indices:
                      new_image_paths.append(image_paths[n_index + 1])
                      new_camera_paths.append(camera_paths[n_index + 1])

                    image_paths = new_image_paths
                    camera_paths = new_camera_paths

                images = [load_image(image_path)for image_path in image_paths]
                ref_width = images[0].width
                ref_height = images[0].height
                images = [resize_image(image, self.image_width, self.image_height) for image in images]
                cameras = [load_camera(camera) for camera in camera_paths]
                intrinsics = [camera[0] for camera in cameras]
                intrinsics = [
                    resize_intrinsics(intrinsic,
                                      self.image_width/4, self.image_height/4,
                                      ref_width, ref_height)
                    for intrinsic in intrinsics
                ]
                extrinsics = [camera[1] for camera in cameras]

                depth_min = cameras[0][4][0]
                depth_max = cameras[0][5][0]
                np_intrinsics = np.stack(intrinsics)
                np_extrinsics = np.stack(extrinsics)
                np_images = np.stack([
                  center_image(np.asarray(image))
                  for image in images
                ])

                np_cams = np.zeros((self.num_neighbors, 2, 4, 4))
                depth_interval = (depth_max - depth_min) / self.depth_interval
                depth_start = depth_min + depth_interval
                depth_end = depth_start + depth_interval * (self.depth_interval - 2)
                np_cams[:, 0] = np_extrinsics
                np_cams[:, 1, 0:3, 0:3] = np_intrinsics
                np_cams[:, 1, 3, 0] = depth_min
                np_cams[:, 1, 3, 1] = depth_interval

                # optionally load depth / normal if mode is training
                if(self.mode == 'train' or self.mode == 'eval'):
                    depth_path = paths['depth']
                    np_depth = resize_depth(load_depth(depth_path), self.image_width/4, self.image_height/4)[:, :, None]
                    masked_depth = mask_depth_image(np_depth, depth_start, depth_end)
                    print(np_depth.min(), np_depth.max(), depth_start, depth_end)
                    yield (np_images, np_cams, masked_depth)
                else:
                    yield (np_images, np_cams)
