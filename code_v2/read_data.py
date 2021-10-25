"""
Read images, and split the image to 5 folds
Then save the data in local file

seven classes for emotion images: 
   0 Angry
   1 Disgust
   2 Fear
   3 Happy 
   4 Neutral
   5 Sad
   6 Surprise
"""
import os
import pandas as pd
import numpy as np
import math

def load_pca(path='data/PCA/SFEW.xlsx'):
    """
    load image pca dataframe
    """

    exl = pd.read_excel(path)
    cols = exl.columns
    exl[cols[0]] = exl[cols[0]].apply(lambda x: x.split('.')[0])
    exl[cols[1]] = exl[cols[1]].apply(lambda x: x - 1)
    
    #rename the colunms
    new_names = ['name', 'class']
    new_names.extend([f'pca{i}' for i in range(10)])
    exl.columns = new_names
    
    return exl


def preprocess_data(path='./data/Images', save_pth = './data/processed'):
    """
    1. read the images_path
    2. merge pca dataframe
    3. divide the data into 5 fold

    expected final df.column:
        name, class, fold,image_path, PCA value, fold
    """

    # class dict
    class_dict = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy' :3,
        'Neutral':4,
        'Sad':5,
        'Surprise':6,
    }

    # load image and convert image to flatten vector
    # create a dataframe to store data
    df = pd.DataFrame(columns=['name', 'class', 'image_path'])

    n_files = 0
    for path,dir_list,file_list in os.walk(path):  
        for file_name in file_list:  
            # ignore the os files
            if file_name == '.DS_Store':
                continue
            
            n_files += 1
            image_path = os.path.join(path, file_name)
            # print(image_path)

            # get image class
            category_name = path.split('/')[-1]
            category =  class_dict[category_name]

            # get image name without the file postfix
            image_name = file_name.split('.')[0]     
            
            df = df.append({'name': image_name, 'class':category, 'image_path':image_path}, ignore_index=True)
            
    # load pca dataframe
    pca = load_pca()

    # keep the consistent dtype for the keys
    df['name'] = df['name'].astype('string')
    pca['name'] = pca['name'].astype('string')
    df['class'] = df['class'].astype('int')
    pca['class'] = pca['class'].astype('int')

    # join df(image) left join the pca if the df.name = pca.name and class
    df_join = pd.merge(df, pca, on=['name', 'class'])
    df_join.dropna(inplace=True)

    # divide into k=5 fold
    df_new = None
    for category in sorted(df['class'].unique()):
        table = df_join.loc[df_join['class'] == category].copy()
        n = table.shape[0]
        n_group = math.ceil(n/5) 
        table = table.sample(frac=1)
        table['rowIndex'] = range(n)
        table['fold'] = table['rowIndex'].apply(lambda x: x//n_group)
        table = table.drop(columns=['rowIndex'])
        if df_new is None:
            df_new = table
        else:
            df_new = df_new.append(table)
    df_new.sort_index(axis=0)

    df_new.to_csv(os.path.join(save_pth,'data.csv'),index = False, header=True)
    print(f'read {df_new.shape[0]} images in total')
    return df_join


def load_dataframe(path = './data/processed/data.csv'):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = preprocess_data()