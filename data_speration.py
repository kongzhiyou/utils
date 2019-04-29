import os, shutil
import numpy as np
import glob

# Path to the full data directory, not categorised into train/val/test sets or category folders
original_dataset_dir = r'E:\a_pinlan_data\金尚互联\0428可乐数据分类的副本'

# The directory where we will store our dataset, divided into train/val/test directories, and further into category directories
base_dir = r'E:\a_pinlan_data\金尚互联\data'

categories = []

dir = glob.glob(r'E:\a_pinlan_data\金尚互联\0428可乐数据分类的副本\*')


for cag in dir:
    a = cag.split('\\')[-1]  #windows
    #a = cag.split('/')[-1] #Linux,MacOs
    categories.append(a)

# We want to keep our data organized into train and validation folders, each with separate category subfolders
str_train_val = ['train', 'validation']

if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    print('Created directory: ', base_dir)

for dir_type in str_train_val:
    train_test_val_dir = os.path.join(base_dir, dir_type)

    if not os.path.exists(train_test_val_dir):
        os.mkdir(train_test_val_dir)

    for category in categories:
        dir_type_category = os.path.join(train_test_val_dir, category)

        if not os.path.exists(dir_type_category):
            os.mkdir(dir_type_category)
            print('Created directory: ', dir_type_category)

directories_dict = {}  # To store directory paths for data subsets.

np.random.seed(12)
for cat in categories:
    list_of_images = np.array(os.listdir(os.path.join(original_dataset_dir,cat)))
    print("{}: {} files".format(cat, len(list_of_images)))
    indexes = dict()
    indexes['validation'] = sorted(np.random.choice(len(list_of_images), size=5, replace=False))
    indexes['train'] = list(set(range(len(list_of_images))) - set(indexes['validation']))
    for phase in str_train_val:
        for i, fname in enumerate(list_of_images[indexes[phase]]):
            source = os.path.join(original_dataset_dir, cat, fname)
            destination = os.path.join(base_dir, phase, cat, str(i)+".jpg")
            shutil.copyfile(source, destination)
        print("{}, {}: {} files copied".format(cat, phase, len(indexes[phase])))
        directories_dict[phase + "_" + cat + "_dir"] = os.path.join(base_dir, phase, cat)