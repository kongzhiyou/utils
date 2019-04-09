# coding = utf-8
import glob
import cv2

'''
jpg图像转png图像(不改变位数)
'''

image_path = r'E:\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages'
save_path = r'E:\labelme\examples\semantic_segmentation\data_dataset_voc\JPEGImages\png'
suffix= '.png'

def image_switch(image_path,save_path):
    image_list = glob.glob(image_path+'/*.jpg')
    for img in image_list:
        image = cv2.imread(img)
        image_name = img.split('\\')[-1].split('.')[0]
        cv2.imwrite(save_path+'\\'+image_name+suffix,image)

if __name__ == '__main__':
    image_switch(image_path,save_path)

