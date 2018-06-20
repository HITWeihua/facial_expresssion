import os
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import re


def write_2_image(lable_vec, image_names_path, image_names, is_flipped=False, angle=0):
    express_array = np.zeros((64, 64, 14))
    index = 0
    for image_name in image_names:
        image_path = os.path.join(image_names_path, image_name)
        img = Image.open(image_path)
        # img = img.convert("L")
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
    express_raw = [float(x) for x in express_array.flatten().tolist()]
    lable_vec = [float(x) for x in lable_vec.tolist()]
    return express_raw


def concat_and_write2file(images_vec, lable_vec):
    img_landmarks_raw = []
    img_landmarks_raw.extend(images_vec)
    # img_landmarks_raw.extend(landmarks_vec)  # 64*64*7+68*2*7=29624
    assert len(img_landmarks_raw) == 57344, "length not equal. img_landmarks_raw: {}".format(len(img_landmarks_raw))
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=lable_vec)),
        'img_landmarks_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img_landmarks_raw))
    }))  # example对象对label和image数据进行封装
    writer.write(example.SerializeToString())  # 序列化为字符串



image_base_path = "../oulu_sampling/"
# lable_base_path = "F:\\files\\joint_fine_tuning\\Emotion_labels\\Emotion"
# landmarks_base_path = "F:\\files\\joint_fine_tuning\\Landmarks\\Landmarks"


# print(people_list)
sampling_number = 7


if __name__ == '__main__':
    init_time = time.time()
    for tf_num in range(10):
        total_samples = 0
        start_time = time.time()

        writer = tf.python_io.TFRecordWriter("../oulu_ld_image_joint/{}/oulu_joint_test_{}.tfrecords".format(str(tf_num), str(tf_num)))  # 要生成的文件

        # with open("./data_pairs/landmark/{}/test_subjects.txt".format(str(tf_num)), 'r') as f:
        #     subject_list = f.readlines()
        people_list = [x for x in os.listdir(image_base_path) if str(tf_num) in x[3]]
        people_list.sort()
        print(people_list)
        for people in people_list:
            # landmarks_people_dir_path = os.path.join(landmarks_base_path, people)
            people_dir_path = os.path.join(image_base_path, people)

            # lable_path_people = os.path.join(lable_base_path, people)
            express_list = [x for x in os.listdir(people_dir_path) if '.DS' not in x]
            express_list.sort()
            print(express_list)
            for num in range(len(express_list)):
                # landmarks_names_path = os.path.join(emotion_people_dir_path, expression)
                image_names_path = os.path.join(people_dir_path, express_list[num])
                # lable_path = os.path.join(lable_path_people, express_list[num])
                # landmark_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.txt' in x]
                image_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and '.jpeg' in x and 'p.jpeg' not in x]
                ld_image_names = [x for x in os.listdir(image_names_path) if '.DS' not in x and 'p.jpeg' in x]
                image_names.sort()
                ld_image_names.sort()

                lable_value = float(num)
                lable_vec = np.zeros((len(express_list)))

                lable_vec[int(lable_value)] = 1
                # print(lable_value)
                # print(lable_vec)

                # landmark_names = average_sampling(landmark_names)
                # angles = [-15, -10, -5, 0, 5, 10, 15]
                # for angle in angles:
                image_names.extend(ld_image_names)
                assert len(image_names) == 14, "files number not equal to 14. files: {}".format(image_names_path)
                images_vec = write_2_image(lable_vec, image_names_path, image_names, is_flipped=False, angle=0)
                concat_and_write2file(images_vec, lable_vec)

                # landmarks_vec = preprocess_data(lable_vec, landmarks_names_path, landmark_names, is_flipped=True, add_noise=False, rotation=False)
                # images_vec = write_2_image(lable_vec, image_names_path, image_names, is_flipped=True, angle=0)
                # concat_and_write2file(images_vec, landmarks_vec, lable_vec)


                total_samples += 1
                # print(total_samples)

        print("fold:{} total_samples:{}".format(tf_num, total_samples))
        print("real cost {}'s".format(time.time() - start_time))
        writer.close()
    print("real total cost {}'s".format(time.time() - init_time))