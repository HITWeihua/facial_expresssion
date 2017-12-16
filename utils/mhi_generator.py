import os
from PIL import Image
import numpy as np
import cv2

image_base_path = "../test/"

sampling_number = 7
threshold = 150
tao = 200
delta = 25


def mhi_generator(image_names_path):
    image_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.jpeg' in x]
    image_number = len(image_names)
    mhi_img = np.zeros((64, 64), np.float32)
    # pre_img = mhi_img
    for i in range(image_number):
        if i == 0:
            image_path = os.path.join(image_names_path, image_names[i])
            img = Image.open(image_path)
            img_array = np.array(img)
            # mhi_img = img_array
            pre_img = img_array
        else:
            image_path = os.path.join(image_names_path, image_names[i])
            img = Image.open(image_path)
            img_array = np.array(img)
            psai = np.abs(img_array - pre_img) > threshold
            new_mhi_img = psai*tao+psai.__invert__()*np.maximum(0, mhi_img-delta)
            mhi_img = new_mhi_img
            pre_img = img_array
    # mhi_img = mhi_img*256/tao
    Image.fromarray(mhi_img).convert("L").save(os.path.join(image_names_path, "mhi.png"), "PNG")


def mhi_of_generator(image_names_path):
    image_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.jpeg' in x]
    image_number = len(image_names)
    mhi_img = np.zeros((64, 64), np.float32)
    # pre_img = mhi_img
    for i in range(image_number):
        if i == 0:
            image_path = os.path.join(image_names_path, image_names[i])
            img = Image.open(image_path)
            img_array = np.array(img)
            # mhi_img = img_array
            pre_img = img_array
        else:
            image_path = os.path.join(image_names_path, image_names[i])
            img = Image.open(image_path)
            img_array = np.array(img)
            psai = np.abs(img_array - pre_img) > threshold

            new_mhi_img = psai * tao + psai.__invert__() * np.maximum(0, mhi_img - delta)
            mhi_img = new_mhi_img
            pre_img = img_array
    # mhi_img = mhi_img*256/tao
    Image.fromarray(mhi_img).convert("L").save(os.path.join(image_names_path, "mhi_of.png"), "PNG")


if __name__ == '__main__':
    people_list = [x for x in os.listdir(image_base_path)]
    for people in people_list:
        people_dir_path = os.path.join(image_base_path, people)
        express_list = [x for x in os.listdir(people_dir_path) if '.DS' not in x]

        for num in range(len(express_list)):
            image_names_path = os.path.join(people_dir_path, express_list[num])

            mhi_generator(image_names_path)
            # mhi_of_generator(image_names_path)
            print('test')