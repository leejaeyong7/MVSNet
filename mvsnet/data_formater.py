from PIL import Image
import os
import numpy as np
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image, W, H):
    np_img = np.array(image.convert('RGB'))
    new_img = cv2.resize(np_img, dsize=(W, H),
                         interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(new_img)

def scale_image(image, scale=0.25):
    np_img = np.array(image.convert('RGB'))
    new_img = cv2.resize(np_img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)
    # pil_img_arr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    return Image.fromarray(new_img)


def scale_intrinsics(intrinsic, scale=0.25):
    intrinsic[0:2, 0:3] *= scale
    return intrinsic

def resize_intrinsics(intrinsic, W, H, OW, OH):
    sx = float(W) / float(OW)
    sy = float(H) / float(OH)

    intrinsic[0:1, 0:3] *= sx
    intrinsic[1:2, 0:3] *= sy
    return intrinsic

def scale_depth(depth, scale=0.25):
    return cv2.resize(depth, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)

def resize_depth(depth, W, H):
    return cv2.resize(depth, dsize=(W, H),interpolation=cv2.INTER_NEAREST)

def scale_normal(normal, scale=0.25):
    scaled_normal = cv2.resize(normal, None,
                               fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
    return scaled_normal / np.linalg.norm(scaled_normal, axis=2)[:, :, None]


def resize_normal(normal, W, H):
    resized_normal = cv2.resize(normal, dsize=(W, H),interpolation=cv2.INTER_NEAREST)
    return resized_normal / np.linalg.norm(resized_normal, axis=2)[:, :, None]

def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    print(img.shape)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def load_image(image_path):
    return Image.open(image_path)


def load_depth(depth_path):
    return np.load(depth_path)


def load_normal(normal_path):
    return np.load(normal_path)


def load_camera(camera_file_path):
    with open(camera_file_path, 'r') as file:
        words = file.read().split()
        # read extrinsic
        intrinsic = np.zeros((3, 3), dtype=np.float32)
        extrinsic = np.zeros((4, 4), dtype=np.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j
                extrinsic[i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 16
                intrinsic[i][j] = words[intrinsic_index]
        words = [w.replace('[', '').replace(']', '') for w in words[-4:]]
        width = np.array([words[-4]], np.float32)
        height = np.array([words[-3]], np.float32)
        min_d = np.array([words[-2]], np.float32)
        max_d = np.array([words[-1]], np.float32)

    return intrinsic, extrinsic, width, height, min_d, max_d
