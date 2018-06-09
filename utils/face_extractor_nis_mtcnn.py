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
base_read_path = os.path.abspath('F:\\files\\人脸识别相片\\人脸识别切割\\174936')
base_write_path = os.path.abspath('F:\\files\\人脸识别相片\\cropped\\174936')

files = [f for f in os.listdir(base_read_path) if '.jpeg' in f]
detector = MTCNN()

for file_num in range(len(files)):
    read_f_path = os.path.join(base_read_path, files[file_num])
    write_f_path = os.path.join(base_write_path, files[file_num])
    print("Processing file: {}".format(read_f_path))
    img = cv2.imread(read_f_path)
    print(detector.detect_faces(img))



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

