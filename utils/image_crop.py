from PIL import Image
import os
import re

pattern = re.compile(r'\_(\d+)\.')
base_path = "F:\\files\\人脸识别相片\\人脸识别"
base_save_path = "F:\\files\\人脸识别相片\\人脸识别切割"
imageset = os.listdir(base_path)
for string in imageset:
    name = re.findall(pattern, string)[0]
    im = Image.open(os.path.join(base_path, string))
    # 图片的宽度和高度
    img_size = im.size
    w = img_size[0] / 4.0
    h = img_size[1] / 4.0
    person_dir = os.path.join(base_save_path, name)
    if not os.path.isdir(person_dir):
        os.mkdir(person_dir)

    for i in range(4):
        for j in range(4):
            x = w*i
            y = h*j
            region = im.crop((x, y, x+w, y+h))
            region.save(os.path.join(person_dir, "crop"+str(i*4+j)+".jpeg"))
