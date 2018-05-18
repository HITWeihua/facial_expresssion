from PIL import Image
import numpy as np
import os


base_path = "F:\\files\\facial_expresssion\\output\\output2"

for i in range(64):
    path = os.path.join(base_path, "conv1_{}.txt".format(i))
    img  = Image.fromarray(np.loadtxt(path)*256)
    # img = img*256
    img.convert('RGB').save(path.replace('.txt', '.jpeg'))