import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np


"""LABEL GENERATION"""
labels = pd.read_csv("WashingtonOBRace/corners.csv",  delimiter = ',', names=['image_name', 'x_top_left', 'y_top_left',
                                                                              'x_top_right', 'y_top_right',
                                                                              'x_bottom_right', 'y_bottom_right',
                                                                              'x_bottom_left', 'y_bottom_left'])

image_name_list = []
for file in os.listdir("WashingtonOBRace/images/"):
    if file.endswith(".png"):
        image_name_list.append(os.path.join(file))

for image_name in image_name_list:
    im = Image.open("WashingtonOBRace/images/" + image_name)
    image_width, image_height = im.size
    matches = labels[labels['image_name'].str.match(image_name)]
    image_name_without_ext = os.path.splitext(image_name)[0]
    label_file_name = image_name_without_ext + '.txt'
    try:
        os.remove('WashingtonOBRace/labels/' + label_file_name)
    except OSError:
        pass
    file = open('WashingtonOBRace/labels/' + label_file_name, 'w')
    for index, row in matches.iterrows():
        # print(row['image_name'], row['x_top_left'])
        x_center = (row['x_top_left'] + row['x_top_right'] + row['x_bottom_right'] + row['x_bottom_left']) / 4 / image_width
        y_center = (row['y_top_left'] + row['y_top_right'] + row['y_bottom_right'] + row['y_bottom_left']) / 4 / image_height
        width = abs((max(row['x_top_right'], row['x_bottom_right']) - min(row['x_top_left'], row['x_bottom_left'])) / image_width)
        height = (max(row['y_bottom_right'], row['y_bottom_left']) - min(row['y_top_right'], row['y_top_left'])) / image_height
        file.write('0 '+ str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
        if width < 0:
            print(image_name)
            print(row)
            print(x_center, y_center, width, height)
            im1 = ImageDraw.Draw(im)
            w, h = 220, 190
            # shape = [(row['x_top_left'], row['y_top_left']), (row['x_top_right'], row['y_top_right'])]
            shape = [(x_center*image_width,y_center*image_height), ((x_center+1/2*width)*image_width, (y_center+1/2*height)*image_height)]
            im1.line(shape, fill="red", width=3)
            im1.point((x_center * image_width, y_center * image_height), fill='green')
            im.show()
            input('wait for keypress')

"""VALIDATION/TRAIN SPLIT"""
validation_split = 32/280
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(image_name_list)
indices = list(range(dataset_size))

val_len = int(np.floor(validation_split * dataset_size))
validation_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(validation_idx))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[val_len:], indices[:val_len]

try:
    os.remove('train.txt')
except OSError:
    pass
file = open('train.txt', 'w')
for i in train_indices:
    file.write('data/custom/images/' + image_name_list[i] + '\n')

try:
    os.remove('valid.txt')
except OSError:
    pass
file = open('valid.txt', 'w')
for i in val_indices:
    file.write('data/custom/images/' + image_name_list[i] + '\n')