import logging
import math
import numpy as np
from mvs_camera import MVSCamera
import os
from os import path

def mkdirp(dest):
    try: 
        os.makedirs(dest)
    except OSError:
        if not path.isdir(dest):
            raise

class MVS:
    def __init__(self, depth_dimension ):
        self.cameras = []
        self.images = []
        self.pair_score_map = None
        self.depth_dimension = depth_dimension


    def from_nvm(self, nvm_object, image_path):
        camera_name_hash = {}
        logging.info('[TRANSLATE MVS] Translation start')
        logging.info('[TRANSLATE MVS] Compute Min / Max / Average distance of cameras begin')

        logging.info('[TRANSLATE MVS] Compute Min / Max / Average distance of cameras finished')
        logging.info('[TRANSLATE MVS] Translating cameras')
        distances = []
        for nvm_point in nvm_object.points:
            for measurement in nvm_point.measurements:
                distance = measurement.camera.get_position().distance_to(nvm_point)
                distances.append(distance)
        avg_dist = np.sum(distances) / len(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        for camera in nvm_object.cameras:
            pos = camera.get_position()
            rot = camera.get_rotation()
            intrinsic = camera.get_intrinsic()
            fx = intrinsic[0,0]
            fy = intrinsic[1,1]
            cx = intrinsic[0,2]
            cy = intrinsic[1,2]


            distances = []
            for index, point in camera.features.iteritems():
                distance = camera.get_position().distance_to(point)
                distances.append(distance)
            if(distances):
                depth_min = np.percentile(distances, 2.5)
                depth_max = np.percentile(distances, 97.5)
            else:
                depth_min = min_dist
                depth_max = max_dist
            depth_int = (depth_max - depth_min) / self.depth_dimension

            # assign index later
            mvs_camera = MVSCamera(0, pos, rot, fx, fy, cx, cy,
                                   depth_min, depth_int)
            camera_name_hash[camera.name] = mvs_camera
            image_file_path = path.join(image_path, '{}.jpg'.format(camera.name))
            mvs_camera.set_image_file(image_file_path)
        for i, nvm_camera_name in enumerate(sorted(camera_name_hash.iterkeys())):
            mvs_camera = camera_name_hash[nvm_camera_name]
            mvs_camera.index = i
            self.cameras.append(mvs_camera)
        num_cameras = len(self.cameras)
        logging.info('[TRANSLATE MVS] finished point cloud')


        # compute score map
        logging.info('[TRANSLATE MVS] Score computation start')
        score_map = np.zeros((num_cameras, num_cameras))
        for i in range(0, num_cameras):
            nvm_camera1 = nvm_object.cameras[i]
            logging.info('[TRANSLATE MVS] On Camera {} / {}'.format(i+1, num_cameras))
            for feature_index, point in nvm_camera1.features.iteritems():
                for camera_name, camera in point.cameras.iteritems():
                    if camera_name == nvm_camera1.name:
                        continue
                    mvs_camera2 = camera_name_hash[camera_name]
                    mvs_camera1 = camera_name_hash[nvm_camera1.name]
                    mvs_camera1_index = mvs_camera1.index
                    mvs_camera2_index = mvs_camera2.index
                    pos_diff1 = mvs_camera1.get_position().to_array() - point.to_array()
                    pos_diff2 = mvs_camera2.get_position().to_array() - point.to_array()
                    pos_abs1 = math.sqrt(np.dot(pos_diff1, pos_diff1))
                    pos_abs2 = math.sqrt(np.dot(pos_diff2, pos_diff2))
                    dot = np.dot(pos_diff1, pos_diff2) / (pos_abs1 * pos_abs2)
                    theta_i_j = (180 / math.pi) * math.acos(dot)
                    score = self.get_score(theta_i_j)
                    score_map[mvs_camera1_index, mvs_camera2_index] += score

        logging.info('[TRANSLATE MVS] Score computation finished')
        self.pair_score_map = score_map

    def get_score(self, theta_i_j, theta_0=5, sigma_1=1, sigma_2=10):
        numerator = -1 * pow((theta_i_j - theta_0), 2)
        if(theta_i_j <= theta_0):
            denominator = 2 * pow(sigma_1, 2)
        else:
            denominator = 2 * pow(sigma_2, 2)
        return math.exp(numerator / denominator)


    def write_to_path(self, output_path, max_pairs = 10):
        # first create necessary paths
        images_path = path.join(output_path, 'images')
        cams_path = path.join(output_path, 'cams')
        pair_file_path = path.join(output_path, 'pair.txt')

        mkdirp(output_path)
        mkdirp(images_path)
        mkdirp(cams_path)

        # write images / cameras
        for camera in self.cameras:
            camera.to_file(output_path)

        # write pairs
        with open(pair_file_path, 'w') as pair_file:
            # first write number of cameras
            num_cameras = len(self.cameras)
            pair_file.write(str(num_cameras))
            pair_file.write('\n')

            # next, for each camera,
            #  write index
            #  on next line, write max_pairs number of best index/scores pair
            for camera in self.cameras:
                cam_index = camera.index

                pair_file.write(str(cam_index))
                pair_file.write('\n')

                cam_scores = self.pair_score_map[cam_index]
                cam_score_index_list = [
                    (score, i)
                    for i, score, in
                    enumerate(cam_scores)
                ]

                cam_score_index_list.sort(reverse=True)
                pair_file.write(str(max_pairs))
                for i in range(0, max_pairs):
                    score_pair = cam_score_index_list[i]
                    score = score_pair[0]
                    index = score_pair[1]

                    pair_file.write(' {} {}'.format(index, score))
                pair_file.write('\n')

