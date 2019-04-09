# coding = utf-8
import random
import glob
import xml.etree.ElementTree as ET
import PIL.Image as Image
import os

'''
将labelImg标注的区域分割出来，并归类到相应的文件夹中
'''

image_path = r'E:\a_pinlan_data\label_data'
image_list = glob.glob(image_path+'\\*.jpg')

for img in image_list:
    xml_name = img.split('.')[0]+'.xml'
    image = Image.open(img)
    if os.path.exists(xml_name):
        with open(xml_name,'r') as f:
            tree = ET.parse(xml_name)
            root = tree.getroot()
            obj_list = root.findall('object')
            for obj in obj_list:
                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                left_up = [int(bndbox.find('xmin').text),int(bndbox.find('ymin').text)]
                right_down = [int(bndbox.find('xmax').text),int(bndbox.find('ymax').text)]
                croped = (left_up[0],left_up[1],right_down[0],right_down[1])
                new_image = image.crop(croped)
                try:

                    if not os.path.exists(image_path+'\\'+obj_name):
                        os.makedirs(image_path+'\\'+obj_name)
                except Exception as e:
                    print(e)
                str = ""
                for i in range(1,8):
                    s = chr(random.randint(97, 122))
                    str = str+s
                try:
                    new_image.save(image_path+'\\'+obj_name+'\\'+obj_name+'_'+str+'.jpg')
                except Exception as e:
                    print(e)