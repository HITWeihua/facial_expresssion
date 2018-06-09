from mtcnn.mtcnn import MTCNN
import cv2
from skimage import io
import os
from skimage.transform import resize
from skimage.color import rgb2gray
import re
import numpy as np


num = 0
spliter = re.compile(r'[\(\)\,\n]+')

# base_read_path = os.path.abspath('F:\\files\\facial_expresssion\\oulu\\Strong\\P013\\Disgust')
# base_write_path = os.path.abspath('../oulu/oulu_face_landmark')
base_read_path = os.path.abspath('/home/duheran/OriginalImg/NI/Strong')
base_write_path = os.path.abspath('/home/duheran/OriginalImg/NI/crop')
person_numbers = os.listdir(base_read_path)
# files = [f for f in os.listdir(base_read_path) if '.jpeg' in f]
detector = MTCNN()

for person in person_numbers:
    read_f_path = os.path.join(base_read_path, person)
    write_f_path = os.path.join(base_write_path, person)
    if not os.path.isdir(write_f_path):
        os.mkdir(write_f_path)
    exps = os.listdir(read_f_path)
    for exp in exps:
        read_i_path = os.path.join(read_f_path, exp)
        write_i_path = os.path.join(write_f_path, exp)
        if not os.path.isdir(write_i_path):
            os.mkdir(write_i_path)
        images = [k for k in os.listdir(read_i_path) if '.jpeg' in k]
        for i in images:
            read_path = os.path.join(read_i_path, i)
            write_path = os.path.join(write_i_path, i)
            print("Processing file: {}".format(read_path))
            img = cv2.imread(read_path)
            box = detector.detect_faces(img)[0]['box']
            img_crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2] , :]
            cv2.imwrite(write_path, img_crop)

# for file_num in range(len(files)):
#     read_f_path = os.path.join(base_read_path, files[file_num])
#     write_f_path = os.path.join(base_write_path, files[file_num])
#     print("Processing file: {}".format(read_f_path))
#     img = cv2.imread(read_f_path)
#     print(detector.detect_faces(img))



    # # RGB
    # if len(img.shape) == 3:
    #     img_crop = img[top:bottom, left:right, :]
    # # gray
    # else:
    #     img_crop = img[top:bottom, left:right]
    #
    #
    # # resize the image to 64*64 and change color from rgb to gray
    # img_resized = resize(img_crop, (64, 64), mode='reflect')
    # if len(img_resized.shape) == 3:
    #     img_gray = rgb2gray(img_resized)
    # else:
    #     img_gray = img_resized
    #
    #
    # # generate 68 tuple landmarks
    # # shape = predictor(img_resized, d)
    # with open(write_f_path.replace('.jpeg', '.txt'), 'w') as fi:
    #     for i in range(shape.num_parts):
    #         fi.write(str(shape.part(i)) + '\n')
    # num += 1
    #
    # io.imsave(write_f_path, img_gray)

