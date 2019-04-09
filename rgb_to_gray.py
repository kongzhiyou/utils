import cv2
import os

'''
将RGB图像转换成GRAY图像(8 bit)
'''

img_path = r'C:\Users\Administrator\Desktop\unet_data\label\label_png'
save_path = r'C:\Users\Administrator\Desktop\unet_data\label\label_png'

if __name__ == "__main__":
    os.chdir(img_path)
    for i in (os.listdir(os.getcwd())):
        img = cv2.imread(img_path+'\\'+i)
        width,height = img.shape[:2][::-1]
        # 检查后缀
        #postfix = os.path.splitext(i)[1]
        #将图片转为灰度图

        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(save_path+'\\'+i,img_gray)