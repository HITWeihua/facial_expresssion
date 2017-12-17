import sys
import os
import dlib
from skimage import io


detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
folder_path = "F:\\files\\facial_expresssion\\oulu\\Strong\\P013\\Disgust"
files = [f for f in os.listdir(folder_path) if '.jpeg' in f]
for f in files:
    f = os.path.join(folder_path, f)
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()