# -*- coding: utf-8 -*-
import cv2
import glob
import json
import PIL.Image as Image
import os
import numpy as np

'''
数据预处理: flip,crop，resize(适用于labelme打标图像)
'''


img_path = 'E:/a_pinlan_data/u_net_data/*.jpg'
root_path = 'E:/a_pinlan_data/u_net_data/'
aug_ops = ['crop']  #翻转、裁剪
crops = [[400,400],[420,420],[440,440],[460,460],[500,500],[480,480]]  #裁剪坐标为[y0:crops[0], x0:crops[1]]
re_size = (256,256)  #resize图片尺寸

#从json文件中提取出一个point,计算机裁剪或者反转,resize后的坐标位置
def caculate_label(point,op,size,i):
    if op=='flip':
        new_w = size[0]-point[0]
        new_h = point[1]
        return [new_w, new_h]
    elif op=='crop':
        if(point[0]<=crops[i][0] and point[1]<=crops[i][1]):
            new_w = point[0]
            new_h = point[1]
            return [new_w, new_h]
        else:
            return None
    elif op=='resize':
        new_w = point[0]*(re_size[0]/size[0])
        new_h = point[1]*(re_size[1]/size[1])
        return [new_w,new_h]

# 遍历json文件，得到所有的点的位置信息,并返回一个新的dict
def get_labels(image_path,op,i,new_name):
    file_name = image_path.split('.')[0]
    with open(file_name+'.json','r') as f:
       parse_list = json.load(f)
       w = parse_list['imageWidth']
       h = parse_list['imageHeight']
       if op=='flip':
           new_w = w
           new_h = h
       elif op=='crop':
           new_w = w if(w<crops[i][0]) else crops[i][0]
           new_h = h if(h<crops[i][1]) else crops[i][1]
       elif op=='resize':
           new_w = re_size[0]
           new_h = re_size[1]
       size = [w,h]
       shapes = parse_list['shapes']
       shape_list = []
       flag = 1
       for shape in shapes:
           point_list = []
           points = shape['points']
           for point in points:
               new_point = caculate_label(point,op,size,i)
               if(new_point!=None):
                   point_list.append(new_point)
               else:flag = 0
           if (flag == 1):
               shape_list.append({'label':shape['label'],'line_color':shape['line_color'],'fill_color':shape['fill_color'],
                                  'points':point_list,'shape_type':shape['shape_type']})
           else:
               flag = 1
               continue
    return {'version':parse_list['version'],'flags':parse_list['flags'],'shapes':shape_list,'lineColor':parse_list['lineColor'],
            'fillColor':parse_list['fillColor'],'imagePath':new_name,'imageData':parse_list['imageData'],'imageHeight':new_h,'imageWidth':new_w}

#将重新定义的标签写到一个新的json文件中
def over_write_lable(labels,op,img_name):
    file_name = img_name.split('.')[0]
    new_file = root_path+op+'/'+file_name+'.json'
    json_data = json.dumps(labels)
    with open(new_file,'a+') as f:
        f.write(json_data)
        f.close()

#实现增强的主要方法，调用各种辅助方法
def augmentor():
    for op in aug_ops:
        img_path_list = glob.glob(img_path)
        for img in img_path_list:
            image = cv2.imread(img)
            img_name = img.split('\\')[1]

            if(op=='flip'):
                new_img = cv2.flip(image,1)
                cv2.imwrite(root_path+op+'/'+img_name,new_img)
                new_name = img_name
                labels = get_labels(img, op,new_name)
                over_write_lable(labels, op,0,img_name)
            elif(op=='crop'):
                #注意：如果图片尺寸小于crops,图片将不会被裁剪
                image_arr = np.array(image)
                h,w,channels = image_arr.shape
                for i in range(len(crops)):
                    if(h>crops[i][1] and w>crops[i][0]):
                        cropped = image[0:crops[i][0],0:crops[i][1]]
                        cv2.imwrite(root_path + op + '/' + img_name.split('.')[0] + '_' + str(i) + '.jpg',cropped)
                        new_name = img_name.split('.')[0] + '_' + str(i) + '.jpg'
                        labels = get_labels(img, op,i,new_name)
                        over_write_lable(labels,op,img_name.split('.')[0]+'_'+str(i)+'.jpg')
# resize图片,输入为图片的根目录
def resize_img(images_path):
    img_list = glob.glob(images_path)
    for image in img_list:
        img = cv2.imread(image)
        new_image = cv2.resize(img,re_size,interpolation=cv2.INTER_AREA) #interpolation是一个插值函数
        img_name = image.split('\\')[-1]
        cv2.imwrite(root_path+'resize'+'/'+img_name,new_image)
        new_name = img_name  #重新生成的图片的名称
        labels = get_labels(image, 'resize',0,new_name)
        over_write_lable(labels, 'resize', img_name)

if __name__ == '__main__':
    #augmentor()
    resize_img(r'E:\a_pinlan_data\u_net_data\crop\*.jpg')

