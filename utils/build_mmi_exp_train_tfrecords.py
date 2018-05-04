import os
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import re


def add_gaussian_noise(landmarks):
    a = np.random.normal(0, 0.01, (len(landmarks)))
    return landmarks+a


def rotaiton_coordinate(landmarks, rotation):
    theata = np.random.random()*(np.pi/5)-(np.pi/10)
    # theata = -rotation * np.pi / 180
    # for i in range(len(landmarks)):
    #     a = np.mat([[np.cos(theata), -np.sin(theata)], [np.sin(theata), np.cos(theata)]]) * np.mat(
    #         (landmarks[i][0] - 32, landmarks[i][1] - 32)).T
    #     landmarks[i] = (float(a[0] + 32), float(a[1] + 32))
    for i in landmarks:
        a = np.mat([[np.cos(theata), -np.sin(theata)], [np.sin(theata), np.cos(theata)]]) * np.mat(i).T
        i = (float(a[0]), float(a[1]))
    return landmarks


def average_sampling(landmark_names):
    landmarks_list_sampling = []
    gap_num = (len(landmark_names) - 2) // (sampling_number - 2)

    landmarks_list_sampling.append(landmark_names[0])
    for i in range(sampling_number - 2):
        landmarks_list_sampling.append(landmark_names[(i + 1) * gap_num])
    landmarks_list_sampling.append(landmark_names[len(landmark_names) - 1])
    return landmarks_list_sampling


def preprocess_data(lable_vec, landmarks_names_path, landmark_names, is_flipped=False, add_noise=False, rotation=False):
    express_array = np.array([])

    for landmark_name in landmark_names:
        landmark_path = os.path.join(landmarks_names_path, landmark_name)
        with open(landmark_path, 'r') as f:
            landmarks = f.readlines()
        landmarks = [re.split(r'[\(\)\,\n]+', x) for x in landmarks]
        landmarks = [(float(x[1]), float(x[2])) for x in landmarks]
        # landmarks = [(round(float(x[0]), 2), round(float(x[1]), 2)) for x in landmarks]
        nose_x = landmarks[30][0]
        nose_y = landmarks[30][1]
        deviation_x, deviation_y = np.std(landmarks, axis=0)  # axis=0计算每一列的标准差
        landmarks = [((x[0]-nose_x)/deviation_x, (x[1]-nose_y)/deviation_y) for x in landmarks]
        if is_flipped:
            landmarks = [(-x[0], x[1]) for x in landmarks]
        if rotation != 0:
            landmarks = rotaiton_coordinate(landmarks, rotation)
        # landmarks = [(round(x[0], 2), round(x[1], 2)) for x in landmarks]
        # landmarks = [(int(x[0]), int(x[1])) for x in landmarks]
        landmarks = np.array(landmarks).flatten()
        if add_noise:
            landmarks = add_gaussian_noise(landmarks)

        express_array = np.hstack((express_array, landmarks))

    landmark_raw = [float(x) for x in express_array.flatten().tolist()]
    lable_vec = [float(x) for x in lable_vec.tolist()]
    return landmark_raw


def preprocess_data_without_01(lable_vec, landmarks_names_path, landmark_names, is_flipped=False, add_noise=False, rotation=False):
    express_array = np.array([])

    for landmark_name in landmark_names:
        landmark_path = os.path.join(landmarks_names_path, landmark_name)
        with open(landmark_path, 'r') as f:
            landmarks = f.readlines()
        landmarks = [re.split(r'[\(\)\,\n]+', x) for x in landmarks]
        landmarks = [(float(x[1]), float(x[2])) for x in landmarks]
        # landmarks = [(round(float(x[0]), 2), round(float(x[1]), 2)) for x in landmarks]
        # nose_x = landmarks[30][0]
        # nose_y = landmarks[30][1]
        # deviation_x, deviation_y = np.std(landmarks, axis=0)  # axis=0计算每一列的标准差
        # landmarks = [((x[0]-nose_x)/deviation_x, (x[1]-nose_y)/deviation_y) for x in landmarks]
        if is_flipped:
            # landmarks = [(-x[0], x[1]) for x in landmarks]
            landmarks = [(64-x[0], x[1]) for x in landmarks]  # -(x[0]-32)+32=-x[0]+64
        if rotation != 0:
            landmarks = rotaiton_coordinate(landmarks, rotation)
        # landmarks = [(round(x[0], 2), round(x[1], 2)) for x in landmarks]
        # landmarks = [(int(x[0]), int(x[1])) for x in landmarks]
        landmarks = np.array(landmarks).flatten()
        if add_noise:
            landmarks = add_gaussian_noise(landmarks)

        express_array = np.hstack((express_array, landmarks))

    landmark_raw = [float(x) for x in express_array.flatten().tolist()]
    lable_vec = [float(x) for x in lable_vec.tolist()]
    return landmark_raw


def write_2_image(lable_vec, image_names_path, image_names, is_flipped=False, angle=0):
    express_array = np.zeros((64, 64, 15))
    index = 0
    for image_name in image_names:
        image_path = os.path.join(image_names_path, image_name)
        img = Image.open(image_path)
        img = img.resize((64, 64))
        img = img.convert('L')
        if is_flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if angle!=0:
            img = img.rotate(angle)
        img_array = np.array(img)
        express_array[:, :, index] = img_array
        index += 1
        # express_array = np.hstack((express_array, img_array.flatten()))
    # express_raw = express_array.tobytes()
    # express_raw = express_array.tostring()
    # lable_vec = lable_vec.tostring()
    express_raw = [float(x / 256) for x in express_array.flatten().tolist()]
    return express_raw


def concat_and_write2file(images_vec, lable_vec):
    img_landmarks_raw = []
    img_landmarks_raw.extend(images_vec)
    # img_landmarks_raw.extend(landmarks_vec)  # 64*64*7+68*2*7=29624
    assert len(img_landmarks_raw) == 61440, "length not equal. img_landmarks_raw: {}".format(len(img_landmarks_raw))
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=lable_vec)),
        'img_landmarks_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img_landmarks_raw))
    }))  # example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  # 序列化为字符串



image_base_path = "../mmi/image"
# lable_base_path = "F:\\files\\joint_fine_tuning\\Emotion_labels\\Emotion"
# landmarks_base_path = "F:\\files\\joint_fine_tuning\\Landmarks\\Landmarks"


# print(people_list)
sampling_number = 15


if __name__ == '__main__':
    init_time = time.time()
    for tf_num in range(10):
        total_samples = 0
        start_time = time.time()

        writer = tf.python_io.TFRecordWriter("../mmi/mmi_images/{}/mmi_images_train_{}.tfrecords".format(str(tf_num), str(tf_num)))  # 要生成的文件

        with open("../mmi/label.txt", 'r') as f:
            label_list = f.readlines()
        label_list = [raw_label.strip().split(" ") for raw_label in label_list]
        label_dict = dict(label_list)
        all_people_list = [x for x in os.listdir(image_base_path)]
        people_list = []
        for i in range(len(all_people_list)):
            if i % 10 == tf_num:
                pass
            else:
                people_list.append(all_people_list[i])

        print(people_list)
        for people in people_list:
            # landmarks_people_dir_path = os.path.join(landmarks_base_path, people)
            people_dir_path = os.path.join(image_base_path, people)

            # lable_path_people = os.path.join(lable_base_path, people)
            express_list = [x for x in os.listdir(people_dir_path) if '.DS' not in x]

            for num in range(len(express_list)):
                # landmarks_names_path = os.path.join(emotion_people_dir_path, expression)
                image_names_path = os.path.join(people_dir_path, express_list[num])
                # lable_path = os.path.join(lable_path_people, express_list[num])
                # landmark_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.txt' in x]
                # image_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.jpg' in x]
                image_names = ['0_0.jpg', '0_3.jpg', '0_6.jpg', '0_8.jpg', '0_10.jpg', '0_12.jpg', '0_14.jpg']
                lable_value = int(label_dict[people+'-'+express_list[num]])
                lable_vec = np.zeros((6))
                lable_vec[lable_value-1] = 1
                # for i in range(len(lable_vec)):
                #     if abs(i+1 - lable_value) < 0.1:
                #         lable_vec[i] = 1
                #     else:
                #         lable_vec[i] = 0
                # print(lable_value)
                # print(lable_vec)

                # landmark_names = average_sampling(landmark_names)
                # angles = [-15, -10, -5, 0, 5, 10, 15]
                # for angle in angles:
                assert len(image_names) == 7, "images files number not equal to 15. files: {}".format(image_names_path)
                angles = [-15, -10, -5, 0, 5, 10, 15]
                for angle in angles:
                    images_vec = write_2_image(lable_vec, image_names_path, image_names, is_flipped=False, angle=angle)
                    concat_and_write2file(images_vec, lable_vec)

                    images_vec = write_2_image(lable_vec, image_names_path, image_names, is_flipped=True, angle=angle)
                    concat_and_write2file(images_vec, lable_vec)

                total_samples += 14
                # print(total_samples)

        print("fold:{} total_samples:{}".format(tf_num, total_samples))
        print("real cost {}'s".format(time.time() - start_time))
        writer.close()
    print("real total cost {}'s".format(time.time() - init_time))