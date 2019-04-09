import cv2
import glob
import random

image_path = '/Users/peter/data/wl/*.jpg'
save_path = '/Users/peter/data/'


'''
将一个文件夹中的很多图片，随机分配到数个文件夹中
'''

def image_divide():
    image_list = glob.glob(image_path)
    for img in image_list:
        num = random.randint(1,len(image_list))
        n = num%3

        if(n==0):i = 1

        elif(n==1):i=2

        else: i=3
        image = cv2.imread(img)
        image_name = img.split('/')[-1]
        cv2.imwrite(save_path+str(i)+'/'+image_name,image)

if __name__ == '__main__':
    image_divide()



