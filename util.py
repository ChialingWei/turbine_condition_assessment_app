from skimage.util import crop
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
from skimage import transform
import numpy as np
import image_similarity as imgsim
import pickle
import os
def crop_img(img, pts_data, buf):
    pt1, pt2, pt3, pt4, w, h = pts_data[0], pts_data[1], pts_data[2], pts_data[3], pts_data[4], pts_data[5]
    min_x = min(pt1['x'], pt2['x'], pt3['x'], pt4['x'])
    max_x = max(pt1['x'], pt2['x'], pt3['x'], pt4['x'])
    min_y = min(pt1['y'], pt2['y'], pt3['y'], pt4['y'])
    max_y = max(pt1['y'], pt2['y'], pt3['y'], pt4['y'])
    resized_img = cv2.resize(img, (w, h))
    crop_y_min, crop_y_max, crop_x_min, crop_x_max = max(0, min_y-buf), max(0, h-max_y-buf), max(0, min_x-buf), max(0, w-max_x-buf)
    crop_img_ori_size = crop(resized_img, ((crop_y_min, crop_y_max), (crop_x_min, crop_x_max), (0, 0)), copy=False)
    crop_img = cv2.resize(crop_img_ori_size, (w, h))
    zoom_crop_y, zoom_crop_x = np.shape(crop_img_ori_size)[0], np.shape(crop_img_ori_size)[1]
    return crop_img_ori_size, crop_img, crop_x_min, crop_y_min, zoom_crop_y, zoom_crop_x

def homography(crop_img, pts_data, crop_x, crop_y, zoom_crop_y, zoom_crop_x):
    pt1, pt2, pt3, pt4, w, h = pts_data[0], pts_data[1], pts_data[2], pts_data[3], pts_data[4], pts_data[5]
    #source coordinates
    src = np.array([(pt1['x']-crop_x)*w/zoom_crop_x, (pt1['y']-crop_y)*h/zoom_crop_y,
                    (pt2['x']-crop_x)*w/zoom_crop_x, (pt2['y']-crop_y)*h/zoom_crop_y,
                    (pt3['x']-crop_x)*w/zoom_crop_x, (pt3['y']-crop_y)*h/zoom_crop_y,
                    (pt4['x']-crop_x)*w/zoom_crop_x, (pt4['y']-crop_y)*h/zoom_crop_y,]).reshape((4, 2))
    #destination coordinates
    dst = np.array([50, 50,
                    50, h-50,
                    w-50, h-50,
                    w-50, 50,]).reshape((4, 2))
    #using skimage’s transform module where ‘projective’ is our desired parameter
    tf = transform.estimate_transform('projective', src, dst)
    tf_img = transform.warp(crop_img, tf.inverse)
    return tf_img

def sequence_crop_image(w_crop, h_crop, img_path, img_save_path):
    crop_coord = {}
    im = cv2.imread(img_path)
    h, w, _ = im.shape
    h_rp = math.ceil(h/h_crop)
    w_rp = math.ceil(w/w_crop)
    count = 0
    img = Image.open(img_path)
    img_arr = np.array(img)
    for h_idx in range(h_rp):
        for w_idx in range(w_rp):
            small_img = img_arr[h_crop*h_idx : h_crop*h_idx+h_crop, w_crop*w_idx : w_crop*w_idx+w_crop]
            image_without_alpha = small_img[:, :, :-1]
            img = Image.fromarray(image_without_alpha)
            if np.shape(img)[0] == w_crop and np.shape(img)[1] == w_crop:
                count += 1
                crop_coord[count] = [h_crop*h_idx, w_crop*w_idx]
                # img.save(f"{img_save_path}/{str(count)}.png")
    with open('crop_coord.pkl', 'wb') as fp:
        pickle.dump(crop_coord, fp)




