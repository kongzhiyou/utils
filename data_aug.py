import math
from PIL import Image,ImageEnhance
import numpy as np
import codecs

train_parameters = {

    "input_size": [3, 224, 224],

    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得

    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得

    "label_dict": {},

    "data_dir": "data/data6912/shinho_0521_245",  # 训练数据存储地址

    "train_file_list": "train.txt",

    "label_file": "label_list.txt",

    "save_freeze_dir": "./shinho-freeze-model",

    "save_persistable_dir": "./persistable-params",

    "continue_train": True,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型

    "pretrained": False,            # 是否使用预训练的模型，对于inceptionv4模型暂无预训练参数

    "pretrained_dir": "data/data6487/ResNet50_pretrained",

    "mode": "train",

    "num_epochs": 120,

    "train_batch_size": 30,

    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值

    "use_gpu": True,

    "image_enhance_strategy": {  # 图像增强相关策略

        "need_distort": True,  # 是否启用图像颜色增强

        "need_rotate": True,   # 是否需要增加随机角度

        "need_crop": True,      # 是否要增加裁剪

        "need_flip": True,      # 是否要增加水平随机翻转

        "hue_prob": 0.5,

        "hue_delta": 18,

        "contrast_prob": 0.5,

        "contrast_delta": 0.5,

        "saturation_prob": 0.5,

        "saturation_delta": 0.5,

        "brightness_prob": 0.5,

        "brightness_delta": 0.125

    },

    "early_stop": {

        "sample_frequency": 50,

        "successive_limit": 100,

        "good_acc1": 0.99

    },

    "rsm_strategy": {

        "learning_rate": 0.002,

        "lr_epochs": [20, 40, 60, 80, 100],

        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]

    },

    "momentum_strategy": {

        "learning_rate": 0.002,

        "lr_epochs": [20, 40, 60, 80, 100],

        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]

    },

    "sgd_strategy": {

        "learning_rate": 0.002,

        "lr_epochs": [20, 40, 60, 80, 100],

        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]

    },

    "adam_strategy": {

        "learning_rate": 0.002

    }

}


def resize_img(img, target_size):
    """

    强制缩放图片

    :param img:

    :param target_size:

    :return:

    """

    target_size = target_size

    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)

    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))

    w = 1. * aspect_ratio

    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w ** 2),

                (float(img.size[1]) / img.size[0]) / (h ** 2))

    scale_max = min(scale[1], bound)

    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,

                                                                scale_max)

    target_size = math.sqrt(target_area)

    w = int(target_size * w)

    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)

    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))

    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)

    return img


def rotate_image(img):
    """

    图像增强，增加随机旋转角度

    """

    angle = np.random.randint(-14, 15)

    img = img.rotate(angle)

    return img


def random_brightness(img):
    """

    图像增强，亮度调整

    :param img:

    :return:

    """

    prob = np.random.uniform(0, 1)

    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']

        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1

        img = ImageEnhance.Brightness(img).enhance(delta)

    return img


def random_contrast(img):
    """

    图像增强，对比度调整

    :param img:

    :return:

    """

    prob = np.random.uniform(0, 1)

    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']

        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1

        img = ImageEnhance.Contrast(img).enhance(delta)

    return img


def random_saturation(img):
    """

    图像增强，饱和度调整

    :param img:

    :return:

    """

    prob = np.random.uniform(0, 1)

    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']

        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1

        img = ImageEnhance.Color(img).enhance(delta)

    return img


def random_hue(img):
    """

    图像增强，色度调整

    :param img:

    :return:

    """

    prob = np.random.uniform(0, 1)

    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']

        delta = np.random.uniform(-hue_delta, hue_delta)

        img_hsv = np.array(img.convert('HSV'))

        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta

        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

    return img


def distort_color(img):
    """

    概率的图像增强

    :param img:

    :return:

    """

    prob = np.random.uniform(0, 1)

    # Apply different distort order

    if prob < 0.35:

        img = random_brightness(img)

        img = random_contrast(img)

        img = random_saturation(img)

        img = random_hue(img)

    elif prob < 0.7:

        img = random_brightness(img)

        img = random_saturation(img)

        img = random_hue(img)

        img = random_contrast(img)

    return img


def custom_image_reader(file_list, data_dir, mode):
    """

    自定义用户图片读取器，先初始化图片种类，数量

    :param file_list:

    :param data_dir:

    :param mode:

    :return:

    """

    with codecs.open(file_list) as flist:
        lines = [line.strip() for line in flist]
