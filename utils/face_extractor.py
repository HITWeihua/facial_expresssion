import sys

import dlib
from skimage import io
import os
from skimage.transform import resize
from skimage.color import rgb2gray

predictor_path = "../shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()
num = 0
# win = dlib.image_window()

base_read_path = os.path.abspath('../oulu/Strong')
base_write_path = os.path.abspath('../oulu/oulu_face')
person_list = os.listdir(base_read_path)

for person in person_list:
    read_path = os.path.join(base_read_path, person)
    write_path = os.path.join(base_write_path, person)
    if not os.path.isdir(write_path):
        os.mkdir(write_path)

    expression_list = os.listdir(read_path)
    for exp in expression_list:
        read_exp_path = os.path.join(read_path, exp)
        write_exp_path = os.path.join(write_path, exp)
        files = [f for f in os.listdir(read_exp_path) if '.jpeg' in f]
        if not os.path.isdir(write_exp_path):
            os.mkdir(write_exp_path)
        face_crop = []
        for file_num in range(len(files)):
            read_f_path = os.path.join(read_exp_path, files[file_num])
            write_f_path = os.path.join(write_exp_path, files[file_num])
            print("Processing file: {}".format(read_f_path))
            img = io.imread(read_f_path)
            # The 1 in the second argument indicates that we should upsample the image
            # 1 time.  This will make everything bigger and allow us to detect more
            # faces.
            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for i, d in enumerate(dets):
                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
                # generate 68 tuple landmarks
                shape = predictor(img, d)
            # print(img.shape)
                if file_num == 0:
                    img_crop = img[d.top():d.bottom(), d.left():d.right(), :]
                    face_crop.append(d.top())
                    face_crop.append(d.bottom())
                    face_crop.append(d.left())
                    face_crop.append(d.right())
                else:
                    img_crop = img[face_crop[0]:face_crop[1], face_crop[2]:face_crop[3], :]

                # resize the image to 64*64 and change color from rgb to gray
                img_resized = resize(img_crop, (64, 64), mode='reflect')
                img_gray = rgb2gray(img_resized)

                # generate 68 tuple landmarks
                # shape = predictor(img_resized, d)
                with open(write_f_path.replace('.jpeg', '.txt'), 'w') as fi:
                    for i in range(shape.num_parts):
                        fi.write(str(shape.part(i)) + '\n')
                num += 1

                io.imsave(write_f_path, img_gray)

print(num)
