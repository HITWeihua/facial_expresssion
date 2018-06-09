"""
    均匀取样
"""

import os
import cv2
import shutil

base_read_path = "../oulu/oulu_face_landmark_rescale"
base_write_path = "../oulu_sampling"
# base_read_path = os.path.abspath('F:\\files\\facial_expresssion\\ck\\extended-cohn-kanade-images\\ck_image_landmark')
# base_write_path = os.path.abspath('F:\\files\\facial_expresssion\\ck\\extended-cohn-kanade-images\\ck_sampling')

people_list = [x for x in os.listdir(base_read_path) if '.DS' not in x and int(x[2:])>13]
# sampling_number = 6

for people in people_list:
    people_read_path = os.path.join(base_read_path, people)

    people_write_path = os.path.join(base_write_path, people)
    if not os.path.isdir(people_write_path):
        os.mkdir(people_write_path)

    exp_dir_list = [x for x in os.listdir(people_read_path) if '.DS' not in x]
    for exp_dir in exp_dir_list:
        exp_read_path = os.path.join(people_read_path, exp_dir)
        exp_write_path = os.path.join(people_write_path, exp_dir)
        if not os.path.isdir(exp_write_path):
            os.mkdir(exp_write_path)
        photo_sampled_list = [x for x in os.listdir(exp_write_path) if '.jpeg' in x and 'p.jpeg' not in x]



        for photo in photo_sampled_list:
            photo_read = os.path.join(exp_read_path, photo.replace('.jpeg', 'p.jpeg'))
            photo_write = os.path.join(exp_write_path, photo.replace('.jpeg', 'p.jpeg'))
            shutil.copy(photo_read, photo_write)
