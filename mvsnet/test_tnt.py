
#!/usr/bin/env python
"""
Copyright 2018, Yao Yao, HKUST.
Training script.
"""

from __future__ import print_function

import os
import time
import sys
import math
import argparse
import numpy as np
from test_dataset import MVSDataset

import cv2
#import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *

# params for datasets
tf.app.flags.DEFINE_string('target_set', None, 
                           """Root path to dense folder.""")
# params for input
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('default_depth_start', 1,
                            """Start depth when training.""")
tf.app.flags.DEFINE_integer('default_depth_interval', 1, 
                            """Depth interval when training.""")
tf.app.flags.DEFINE_integer('max_d', 192, 
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 1152, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 864, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 0.8, 
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_integer('base_image_size', 32, 
                            """Base image size to fit the network.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """training batch size""")

# params for config
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
                           '/home/ubuntu/Research/MVSNet/model/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 70000,
                            """ckpt step.""")
FLAGS = tf.app.flags.FLAGS

def mvsnet_pipeline(dataset_dir):
    """ mvsnet in altizure pipeline """
    output_root = '/data/outputs/eth3d/mvsnet/'
    output_dir = os.path.join(output_root, FLAGS.target_set)
    output_folder = os.path.join(output_dir,'depths_mvsnet')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Training generator
    dataset = MVSDataset(dataset_dir, FLAGS.view_num, FLAGS.max_d, FLAGS.max_w, FLAGS.max_h, FLAGS.target_set)
    mvs_generator = iter(dataset)
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.int32)
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)
    
    # iterators
    mvs_iterator = mvs_set.make_initializable_iterator()
    
    # data
    test = mvs_iterator.get_next()
    print(len(test))
    stride_images, input_images, input_cams, image_index = mvs_iterator.get_next()
    stride_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    input_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    input_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(input_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(input_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

    # depth map inference
    init_depth_map, prob_map = inference_mem(stride_images, input_cams, FLAGS.max_d, depth_start, depth_interval)

    # refinement 
    ref_image = tf.squeeze(tf.slice(stride_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    depth_map = depth_refine(init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval)
                                            
    # init option
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()
    # GPU grows incrementally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 2.0
    # config.gpu_options.experimental.use_unified_memory = True

    with tf.Session(config=config) as sess:   

        # initialization
        sess.run(var_init_op)
        sess.run(init_op)
        total_step = 0

        # load model
        if FLAGS.pretrained_model_ckpt_path is not None:
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(
                sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
            print(Notify.INFO, 'Pre-trained model restored from %s' %
                  ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            total_step = FLAGS.ckpt_step
    
        # run inference for each reference view
        sess.run(mvs_iterator.initializer)
        step = 0
        while True:

            start_time = time.time()
            try:
                out_depth_map, out_init_depth_map, out_prob_map, out_images, out_cams, out_index = sess.run(
                    [depth_map, init_depth_map, prob_map, input_images, input_cams, image_index])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (step, duration), 
                  Notify.ENDC)

            # squeeze output
            out_estimated_depth_image = np.squeeze(out_depth_map)
            out_init_depth_image = np.squeeze(out_init_depth_map)
            out_prob_map = np.squeeze(out_prob_map)
            out_ref_image = np.squeeze(out_images)
            out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
            out_ref_cam = np.squeeze(out_cams)
            out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
            out_index = np.squeeze(out_index)

            # paths
            depth_map_path = output_folder + ('/%08d.pfm' % out_index)
            init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
            prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
            out_ref_image_path = output_folder + ('/%08d.png' % out_index)
            out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)

            # save output
            write_pfm(init_depth_map_path, out_init_depth_image)
            write_pfm(depth_map_path, out_estimated_depth_image)
            write_pfm(prob_map_path, out_prob_map)
            out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
            image_file = file_io.FileIO(out_ref_image_path, mode='w')
            scipy.misc.imsave(image_file, out_ref_image)
            write_cam(out_ref_cam_path, out_ref_cam)
            total_step += 1
            step += 1


def main(_):  # pylint: disable=unused-argument
    # mvsnet inference
    mvsnet_pipeline('/mnt/data/datasets/eth3d/train/sets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_set', type=str, default = FLAGS.target_set)
    parser.add_argument('--view_num', type=int, default = FLAGS.view_num)
    args = parser.parse_args()
    FLAGS.target_set= args.target_set
    FLAGS.view_num = args.view_num
    print ('Testing MVSNet with %d views' % args.view_num)
    tf.app.run()
