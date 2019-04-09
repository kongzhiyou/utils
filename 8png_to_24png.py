import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import sys
import shutil

'''
8位图像转24位图像
'''

path = 'E:\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG'
newpath = 'E:\labelme\examples\semantic_segmentation\data_dataset_voc\SegmentationClassPNG\png'


def turnto24(path):
    fileList = []
    files = glob.glob(path+'\\*.png')
    i = 0
    for f in files:
        img = Image.open(f).convert('RGB')
        dirpath = newpath
        file_name, file_extend = os.path.splitext(f)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.png')
        img.save(dst)


turnto24(path)
