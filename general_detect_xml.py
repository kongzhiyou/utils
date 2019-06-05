#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用商品检测引擎 API
"""
import sys
import cv2
import os
import numpy as np
import tensorflow as tf
import urllib.request
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# MODEL_URL = 'https://pinlandata.oss-cn-hangzhou.aliyuncs.com/shinho-models/retinanet_inference_v3.h5?'              'OSSAccessKeyId=LTAITpP5run2spu9&Expires=37537257342&Signature=zHYv1W0Sa9MFtNwT3J0Zh22Vfqs%3D'
#MODEL_URL = 'https://pinlandata.oss-cn-hangzhou.aliyuncs.com/%E4%BB%8A%E9%BA%A6%E9%83%8E/jinmailang_0414.h5'  
#MODEL_URL = 'http://pinlandata.oss-cn-hangzhou.aliyuncs.com/pinshi-models%2Fretinanet_inference_weiquan.h5?OSSAccessKeyId=LTAIqKQXqRiVXNjj&Expires=1557377126&Signature=%2BIM%2Bj2ts%2BJdh%2FpJ8AfMbAW%2B%2BlaM%3D'
# CLASSES = {0: 'bottle', 1: 'jar', 2: 'bucket', 3: 'box', 4: 'bag', 5: 'tanzi', 6: 'others'}
#CLASSES = {0: 'bag', 1: 'noodle bucket', 2: 'package'}
#CLASSES = {0: 'package', 1: 'bag', 2: 'noodle bucket'}
CLASSES = {0: 'bottle', 1: 'jar', 2: 'bucket', 3: 'box', 4: 'bag', 5: 'tanzi', 6: 'others', 7: 'combination'}
'''
MODEL_URL = 'https://pinlandata.oss-cn-hangzhou.aliyuncs.com/pinshi-models/retinanet_inference_weiquan.h5?' \
            'OSSAccessKeyId=LTAISYlYdNBdcBI1&Expires=2623826974&Signature=wG4t8UZoU%2FByciOurrGDBh5sH5s%3D'
CLASSES = {0: 'bottle', 1: 'jar', 2: 'bucket', 3: 'box', 4: 'bag', 5: 'tanzi', 6: 'others', 7: 'combination'}
'''
SCORE_THRESHOLD = .5


# In[3]:


def schedule(blocknum, blocksize, totalsize):
    recv_size = blocknum * blocksize
    f = sys.stdout
    pervent = recv_size / totalsize
    percent_str = "%.2f%%" % (pervent * 100)
    n = round(pervent * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(percent_str.ljust(8, ' ') + '[' + s + ']')
    f.flush()
    f.write('\r')

#求矩形框面积
def bbox_area(bbox):
    cal_area = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    bbox_area = cal_area(bbox)
    return bbox_area

#返回bbox1与bbox2相交面积除以bbox1的面积，和bbox1与bbox2交集矩形框的坐标
def Iou_temp(bbox1,bbox2):
    intersect_min = list((max(bbox1[i], bbox2[i]) for i in range(0, 2)))
    intersect_max = list((min(bbox1[i], bbox2[i]) for i in range(2, 4)))

    intersect_bbox = intersect_min + intersect_max

    intersect_wh = list((max(intersect_max[i] - intersect_min[i], 0) for i in range(2)))
    intersect_area = intersect_wh[0] * intersect_wh[1]
    bbox1_area = bbox_area(bbox1)

    return intersect_area / bbox1_area, intersect_bbox 


#求列表iou_bbox_list中矩形的并集的面积,降低精度要求，坐标整数化
def Union_arae(iou_bbox_list):
    int_bbox_list = []
    for item in iou_bbox_list:
        int_bbox_list.append(list(map(int,item)))

    x = []
    y = []

    for item in int_bbox_list:
        y.append(item[0])
        y.append(item[2])
        x.append(item[1])
        x.append(item[3])
    
    xmax = max(x)
    ymax = max(y)

    temp = np.zeros((ymax+1,xmax+1))

    for item in int_bbox_list:
        for i in range(item[0],item[2]+1):
            for j in range(item[1],item[3]+1):
                temp[i][j] = 1

    union_area = np.sum(temp == 1) 

    return union_area


class GeneralDetectLib(object):
    def __init__(self):
        self.class_list = CLASSES
        self.score_thresh = SCORE_THRESHOLD
        self.graph = tf.get_default_graph()
        #self.model = models.load_model(self.get_model(), backbone_name='resnet50')
        self.model = models.load_model('retinanet_inference_weiquan', backbone_name='resnet50')


    @staticmethod
#     def get_model():
#         model_saved_path = os.path.join(
#             os.path.dirname(os.path.dirname(os.path.abspath('__file__'))),
#             "models/detection_model")
#         final_model_path = os.path.join(model_saved_path, "jinmailang_0414.h5")
#         #final_model_path = os.path.join(model_saved_path, "retinanet_inference_mars.h5")
#         if os.path.exists(final_model_path):
#             print("[Note] General detection model already exists, skip downloading...")
#             return final_model_path
#         if not os.path.exists(model_saved_path):
#             os.makedirs(model_saved_path)
#         print("[Note] General detection model does not exist, downloading...")
#         urllib.request.urlretrieve(MODEL_URL, final_model_path, schedule)
#         print("[Note] General detection model downloaded...")

#         return final_model_path
    
    def is_useless(self,bbox,bbox_list):
        iou_bbox_list = []

        for b in bbox_list:
            if np.equal(bbox,b).all():
                continue
            
            iou = Iou_temp(bbox,b)[0]
            if iou == 1:
                return True
            elif iou >0.2:
                iou_bbox_list.append(Iou_temp(bbox,b)[1])
             
        if len(iou_bbox_list) == 0:
            return False

        if Union_arae(iou_bbox_list)/bbox_area(bbox) >= 0.95:
            return True
        
        return False


    def detect(self, image_obj):
        """
        通用检测对外接口
        :param image_obj: 待检测图像, Pillow Image Object 类型
        :return: 检测的 bounding boxes 结果, 维度: [n_boxes, 6], 第二维: [ymin, xmin, ymax, xmax, class, score]
        """
        image_arr = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)
        image_arr = preprocess_image(image_arr)
        image_arr, scale = resize_image(image_arr)

        with self.graph.as_default():
            boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image_arr, axis=0))

        # correct for image scale
        boxes /= scale

        num = [scores[0][i] for i in range(len(scores[0])) if scores[0][i] >= self.score_thresh]
        num = len(num)

        results = []

        for box, score, label in zip(boxes[0][:num], scores[0][:num], labels[0][:num]):
            '''
            # scores are sorted so we can break
            if score < self.score_thresh:
                break
            '''
            
            if self.is_useless(box,boxes[0][:num]):
                continue
            

            box = list(map(int, box))
            score = round(score, 2)
            results.append([box[1], box[0], box[3], box[2], self.class_list[label], score])
#             results.append([box[1], box[0], box[3], box[2], self.class_list[0], score])
                        
        
        return results


# In[4]:


if __name__ == '__main__':
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import glob
    import pascal_voc_io

    images = glob.glob(os.path.join('/mnt/weiquan' ,'*.jpg'))
    foldername = '/mnt/weiquan_xml'
    
    detector = GeneralDetectLib()

    for image in images:
        img_obj = Image.open(image)
        general_res = detector.detect(img_obj)
        img = cv2.imread(image)       
        
        filename = os.path.splitext(image)[0]
        imgSize = img.shape
        pascal = pascal_voc_io.PascalVocWriter(foldername, filename, imgSize)
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        for item in general_res:
            cv2.rectangle(img,(item[1], item[0]), (item[3], item[2]), (255, 0, 0), 2)
            cv2.putText(img, item[4], (item[1], item[0]), font, 0.7, (0, 255, 0), 2)
            pascal.addBndBox(item[1], item[0], item[3], item[2], item[4], difficult=1)
        pascal.save()
        cv2.imwrite('/mnt/weiquan_xml' + os.path.basename(image),img)
        #cv2.imwrite(image + 'new.jpg',img)


# In[ ]:




