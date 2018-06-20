from PIL import Image
import numpy as np
import os


base_path = "/home/duheran/facial_expresssion/prcv_expreiment/trainging/summaries_new/test_imgs/0/"

for i in range(7):
    path = os.path.join(base_path, "test_{}.txt".format(i))
    img  = Image.fromarray(np.loadtxt(path))
    # img = img*256
    img.convert('RGB').save(path.replace('.txt', '.jpeg'))