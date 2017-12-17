"""
    均匀取样
"""

import os
import cv2
import shutil

base_read_path = "../oulu/oulu_face_landmark"
base_write_path = "../oulu_sampling"

people_list = [x for x in os.listdir(base_read_path) if '.DS' not in x]
sampling_number = 7

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
        photo_list = [x for x in os.listdir(exp_read_path) if '.jpeg' in x]

        photo_list_sampling = []
        gap_num = (len(photo_list)-2)//(sampling_number-2)

        photo_list_sampling.append(photo_list[0])
        for i in range(sampling_number-2):
            photo_list_sampling.append(photo_list[(i+1)*gap_num])
        photo_list_sampling.append(photo_list[len(photo_list)-1])

        for photo in photo_list_sampling:
            photo_read = os.path.join(exp_read_path, photo)
            photo_write = os.path.join(exp_write_path, photo)
            shutil.copy(photo_read, photo_write)

            landmarks = photo.replace('.jpeg', '.txt')
            landmarks_read = os.path.join(exp_read_path, landmarks)
            landmarks_write = os.path.join(exp_write_path, landmarks)
            shutil.copy(landmarks_read, landmarks_write)
            # image = cv2.imread(photo_path)
            # cv2.imwrite(photo_write, image)
