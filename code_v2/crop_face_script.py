from os import read
import cv2
import face_recognition
import pandas as pd
import numpy as np
import read_data
from matplotlib import pyplot as plt
from PIL import Image
import os

df = read_data.load_dataframe()
detected_false = 0
to_path = 'data/crop_faces'
final_path_list = []
for index, row in df.iterrows():
    # read image
    name = row['name']
    label = row['class']
    image_path = row['image_path']

    # detect face
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    # some iamge may not detect face, for false detected images, just keep the original data
    if len(face_locations) > 0:
        (y, x, y1, x1) = face_locations[0]
        face = image[y:y1, x1:x, :]
    else:
        detected_false += 1
        face = image
    
    # resize image to 256, 256, 3
    face = Image.fromarray(face)
    face = face.resize((256,256))
    dir_path = os.path.join(to_path, str(label))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('made directorary ', dir_path)
    
    # save to data/crop_faces/[class]/name.img
    save_path = os.path.join(dir_path, '{}.png'.format(name))
    face.save(save_path)

    df.loc[index, 'crop_path'] = save_path
    df.loc[index, 'croped'] = str(len(face_locations) > 0)
    

df.to_csv('./data/processed/data.csv',index = False, header=True)
print(detected_false)
