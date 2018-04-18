import os
import dlib
import glob
import numpy as np
from skimage import io


predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
base_faces_folder_path = "/home/duheran/facial_expresssion/nis"

detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
scores_matrix = []

def cal_person_id(faces_folder_path):
    img_list = os.listdir(faces_folder_path)
    train_img_list = img_list[:2]
    face_list = np.zeros((2, 128))
    # Now process all the images
    for i in range(len(train_img_list)):
        print("Processing file: {}".format(os.path.join(faces_folder_path, train_img_list[i])))
        img = io.imread(os.path.join(faces_folder_path, train_img_list[i]))

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        rects = dlib.rectangles()
        rects.extend([d.rect for d in dets])

        # Now process each face we found.
        for k, d in enumerate(rects):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. Here we just print
            # the vector to the screen.
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            face_list[i] = list(face_descriptor)
            # print(face_list)
    return face_list


def cal_scores(img_path):
    img = io.imread(img_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))

    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])

    # Now process each face we found.
    for k, d in enumerate(rects):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        return list(face_descriptor)


def predict_preson(img_scores, person):
    predict_name = ""
    ecu_score = {}
    for key, value in person.items():
        score = [0, 0]
        for i in range(len(value)):
            score[i] = np.sqrt(np.sum((np.array(img_scores) - value[i]) ** 2))
            # score[i] = np.linalg.norm(np.array(img_scores) - value[i])
        ecu_score[key] = min(score)
    predict_person = min(zip(ecu_score.values(), ecu_score.keys()))
    scores_matrix.append(sorted(ecu_score.values()))
    if predict_person[0] >= 0.6:
        return predict_name
    else:
        return predict_person[1]


if __name__ == '__main__':
    name_list = sorted(os.listdir(base_faces_folder_path))
    train_name_list = name_list[:80]
    test_name_list = name_list[80:]
    person = {}
    total_correct = 0
    cannot_detect = []
    for name in train_name_list:
        person[name] = cal_person_id(os.path.join(base_faces_folder_path, name))
    print("calculate id finished")
    for name in name_list:
        person_path = os.path.join(base_faces_folder_path, name)
        pictures = os.listdir(person_path)
        if name in train_name_list:
           pictures = pictures[2:]
        if name in test_name_list:
            print(name)
        for img in pictures:
            img_scores = cal_scores(os.path.join(person_path, img))
            if img_scores:
                predict_name = predict_preson(img_scores, person)
                if name in train_name_list and name == predict_name:
                    total_correct += 1
                elif name in test_name_list and name == "":
                    total_correct += 1
            else:
                cannot_detect.append(os.path.join(person_path, img))

    print(total_correct/1760)#1760)
    print(cannot_detect)
    print(len(cannot_detect))
    with open('scores_matrix.txt', 'w') as f:
        for sm in scores_matrix:
            f.write(str(sm))
            f.write("\n")
    # print(scores_matrix)