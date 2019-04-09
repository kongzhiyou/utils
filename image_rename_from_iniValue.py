# coding = utf-8
import cv2
import glob

'''
批量图像重命名
'''

image_path = r'C:\Users\Administrator\Desktop\unet_data\image8'
save_path = r'C:\Users\Administrator\Desktop\unet_data\image8\image_png'

def image_rename():
    image_list = glob.glob(image_path+'/*.png')
    i = 0
    for img in image_list:
        image = cv2.imread(img)
        suffix = img.split('.')[1]
        a = save_path + '\\' + str(i) + '.' + suffix
        cv2.imwrite(save_path + '\\' + str(i) + '.' + suffix, image)
        i = i + 1

if __name__ == '__main__':
    image_rename()



