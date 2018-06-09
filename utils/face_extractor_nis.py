import sys

import dlib
from skimage import io
import os
from skimage.transform import resize
from skimage.color import rgb2gray
import re
import numpy as np

predictor_path = "../shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()
num = 0
spliter = re.compile(r'[\(\)\,\n]+')
# win = dlib.image_window()

# base_read_path = os.path.abspath('../oulu/Strong')
# base_write_path = os.path.abspath('../oulu/oulu_face_landmark')
base_read_path = os.path.abspath('F:\\files\\人脸识别相片\\人脸识别切割\\181808')
base_write_path = os.path.abspath('F:\\files\\人脸识别相片\\cropped\\181808')

files = [f for f in os.listdir(base_read_path) if '.jpeg' in f]

for file_num in range(len(files)):
    read_f_path = os.path.join(base_read_path, files[file_num])
    write_f_path = os.path.join(base_write_path, files[file_num])
    print("Processing file: {}".format(read_f_path))
    img = io.imread(read_f_path)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) != 1:
        pass
        # error_path.append((read_f_path, len(dets)))
    # assert len(dets) == 1, "face number not equal to 1. files: {}".format(read_f_path)
    for i, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        # generate 68 tuple landmarks
        shape = predictor(img, d)
    # print(img.shape)
        landmarks = []
        for i in range(shape.num_parts):
            landmarks.append(spliter.split(str(shape.part(i))))
        # landmarks = [spliter.split(x) for x in shape.part]
        landmarks = [(float(x[1]), float(x[2])) for x in landmarks]
        landmarks = np.array(landmarks)
        # print(list(str(shape.part(0))))
        top = int(max(np.min(landmarks[:, 1])-20, 0))
        bottom = int(np.max(landmarks[:, 1]))

        right = int(np.max(landmarks[:, 0]))
        left = int(np.min(landmarks[:, 0]))

        # img_crop = img[d.top():d.bottom(), d.left():d.right(), :]

        # RGB
        if len(img.shape) == 3:
            img_crop = img[top:bottom, left:right, :]
        # gray
        else:
            img_crop = img[top:bottom, left:right]


        # resize the image to 64*64 and change color from rgb to gray
        img_resized = resize(img_crop, (64, 64), mode='reflect')
        if len(img_resized.shape) == 3:
            img_gray = rgb2gray(img_resized)
        else:
            img_gray = img_resized


        # generate 68 tuple landmarks
        # shape = predictor(img_resized, d)
        with open(write_f_path.replace('.jpeg', '.txt'), 'w') as fi:
            for i in range(shape.num_parts):
                fi.write(str(shape.part(i)) + '\n')
        num += 1

        io.imsave(write_f_path, img_gray)

