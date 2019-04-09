#coding:utf-8
import os
from PIL import Image

'''
24位图像转8位图像(灰度)
'''

#源目录
MyPath = r'C:\Users\Administrator\Desktop\unet_data\image'
#输出目录
OutPath = r'C:\Users\Administrator\Desktop\unet_data\image8'

def processImage(filesoure, destsoure, name, imgtype):
    '''
    filesoure是存放待转换图片的目录
    destsoure是存在输出转换后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'jpg'

    #打开图片
    im = Image.open(filesoure +'\\'+ name)
    #操作转8bit
    width = im.size[0]
    height = im.size[1]
    for x in range(width):
      for y in range(height):
        print(im.getpixel((x,y)))
        if(im.getpixel((x, y))>0):
           im.putpixel((x, y), 1)
        else:
            im.putpixel((x, y), 0)
    im.save(destsoure + name, imgtype)
def run():
    #切换到源目录，遍历源目录下所有图片
    os.chdir(MyPath)
    for i in os.listdir(os.getcwd()):
        #检查后缀
        postfix = os.path.splitext(i)[1]
        if  postfix == '.jpg':
            processImage(MyPath, OutPath, i, postfix)

if __name__ == '__main__':
    run()