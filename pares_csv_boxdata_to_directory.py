import csv
import os
import PIL.Image as Image
import random
import xml.etree.ElementTree as ET
from xml.etree.cElementTree import ElementTree,Element

file_path = '/Users/peter/data/AI/training_annotation_20190409.csv'
root_path = '/Users/peter/data/AI'
image_path = '/Users/peter/data/AI/train'
annotation_path = '/Users/peter/data/AI/annotation'

def read_excle():

    message_list = csv.reader(open(file_path))
    i = 0
    for msg in message_list:
        if not i==0:
            image_name = msg[0]
            x1,y1,x2,y2 = msg[1:-1]
            class_name = msg[len(msg)-1]
            class_path = root_path+'/'+class_name
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            croped = (int(x1),int(y1),int(x2),int(y2))
            image = Image.open(image_path+'/'+image_name)
            crop_iamge = image.crop(croped)
            str = ""
            for i in range(0, 8):
                s = chr(random.randint(97, 122))
                str = str + s
            new_image_name = root_path+'/'+class_name+'/'+class_name+'_'+str+'.jpg'
            try:
                crop_iamge.save(new_image_name)
                create_xmlFile(image_name,msg)
            except Exception as e:
                print(e)

        i = i + 1

def create_xmlFile(image_name,message):

    xml_path = annotation_path+'/'+image_name.split('.')[0]+'.xml'
    if not os.path.exists(xml_path):
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write()
        root = Element("annotation")
        file_name = Element('file_name')
        file_name.attrib = message[0]
        root.append(file_name)
        size = Element('size')
        root.append(size)
        width = Element('width')
        height = Element('height')
        depth = Element('depth')
        size.append(width)
        size.append(height)
        size.append(depth)
    node_object = Element('object')
    name = Element('name')
    name.attrib = message[len(message)-1]
    node_object.append(name)
    bndbox = Element('bndbox')
    xmin = Element('xmin')
    xmin.attrib = message[1]
    ymin = Element('ymin')
    ymin.attrib = message[2]
    xmax = Element('xmax')
    xmax.attrib = message[3]
    ymax = Element('ymax')
    ymax.attrib = message[4]
    bndbox.append(xmin)
    bndbox.append(ymin)
    bndbox.append(xmax)
    bndbox.append(ymax)
    node_object.append(bndbox)
    root.append(node_object)
    tree = ET.ElementTree(root)
    tree.write(xml_path, 'utf-8')

if __name__ == '__main__':
    read_excle()
    create_xmlFile()
