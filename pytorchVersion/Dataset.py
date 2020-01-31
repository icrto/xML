import torch
import numpy as np
import cv2
import os
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from skimage.util import random_noise
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, preprocess=utils.norm, weakly=False, img_size=(224, 224), augmentation=False, aug_prob=0.8):
        super(Dataset, self).__init__()
        self.df = df
        self.len = len(self.df)
        self.preprocess = preprocess
        self.weakly = weakly
        self.img_size = img_size
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        
    def __getitem__(self, index):
        img_name = self.df.iloc[index]['imageID']
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if(img.shape != self.img_size):
            img = cv2.resize(img, self.img_size) 
       
        label = torch.tensor(self.df.iloc[index]['label'], dtype=torch.int64)
        mask = []

        if(self.weakly):
            mask_name = self.df.iloc[index]['maskID']
            mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)  
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            if(mask.shape != self.img_size):
                mask = cv2.resize(mask, self.img_size) 
            
        if(self.augmentation):
            aug = strong_aug(p=self.aug_prob)
            
            if(self.weakly):
                augmented = aug(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                #cv2.imwrite('img' + str(index) + '.png', img)
                #cv2.imwrite('mask' + str(index) + '.png', mask)
            else:
                augmented = aug(image=img)
                img = augmented['image']
        
        img = self.preprocess(img)

        img = img.transpose(2, 1, 0)
        img = torch.tensor(img, dtype=torch.float32)
        
        if(self.weakly):
            mask = self.preprocess(mask)
            mask = mask.transpose(2, 1, 0)
            mask = torch.tensor(mask, dtype=torch.float32)

        
        return img, label, img_name.split('/')[-1], mask
 
    def __len__(self):
        return self.len



def load_synthetic_dataset(folder, dataset='simplified_no_colour', test=False, weakly=False, class_weights=None):
    test_df = None
    if(dataset == 'simplified_no_colour'):
        dtset_folder = os.path.join(folder, 'Synthetic Dataset/simplified_no_colour')
        if(test):
            test_dtset_folder = os.path.join(folder, 'Synthetic Dataset/simplified_no_colour_test')
            test_df = pd.read_excel(os.path.join(test_dtset_folder, 'data.xlsx'), index_col=None)
            test_df['imageID'] = [os.path.join(test_dtset_folder, datum) for datum in test_df.imageID.values]
            if(weakly):
                test_df['maskID'] = [datum[:-4] + '_mask.jpg' for datum in test_df.imageID.values]

    df = pd.read_excel(os.path.join(dtset_folder, 'data.xlsx'), index_col=None)
    df['imageID'] = [os.path.join(dtset_folder, datum) for datum in df.imageID.values]
    if(weakly):
        df['maskID'] = [datum[:-4] + '_mask.jpg' for datum in df.imageID.values]
    
    weights = compute_class_weight(class_weights, np.unique(df.label.values), df.label.values)

    tr_sessions, val_sessions = train_test_split(df.index.values, stratify=df.label.values, random_state=42)
    tr_df = df.loc[df.index.isin(tr_sessions)]
    val_df = df.loc[df.index.isin(val_sessions)]


    print(len(tr_df), len(val_df))
    return tr_df, val_df, test_df, weights

def load_NIH_NCI(folder, weakly=False, cropped=True, class_weights=None):
    img_path = os.path.join(folder, 'DATASETS/NIH-NCI')
    dft = pd.DataFrame()
    for subpath in ['ALTS', 'Biopsy', 'CVT', 'NHS']:
        df = pd.read_excel(os.path.join(img_path, 'data', subpath, 'covariate_data.xls'), header=4)
    
        for ft, missing in [('WRST_HIST_AFTER_DT', '.'), ('HPV_DT', '.'),
                            ('AGE_GRP', '-1'), ('HPV_STATUS', '-1'),
                            ('WRST_HIST_AFTER', -2)]:
            df.loc[df[ft].astype(str) == missing, ft] = np.nan
            df[ft] = df[ft].astype(np.float)
    
        for i in np.arange(df.shape[0]):   
            row = df.iloc[i]  
            if(weakly or not cropped): 
                df.loc[i, 'path'] =  os.path.join(img_path, 'data2', subpath, row.GG_IMAGE_ID)              
            elif(cropped):
                df.loc[i, 'path'] =  os.path.join(img_path, 'cropped', subpath, row.GG_IMAGE_ID)

        dft = pd.concat([dft, df])
            
    dft = dft.dropna(subset=['WRST_HIST_AFTER'])
    df_ = pd.DataFrame({'label': [0 if(int(hist) <= 1) else 1 for hist in dft.WRST_HIST_AFTER]})
    df_['imageID'] = [p for p in dft.path]
    if(weakly):
        df_['maskID'] = [p[:-4] + '_mask.jpg' for p in dft.path]
    df_['sessionID'] = [index for index, _ in enumerate(dft.GG_PATIENT_ID)]

    weights = compute_class_weight(class_weights, np.unique(df_.label.values), df_.label.values)

    session_df = df_[['sessionID', 'label']].groupby('sessionID').agg('max')

    # Stratified partitioning by ground-truth
    train_sessions, test_sessions, _, _ = train_test_split(session_df.index.values,
                                                    session_df.label.values,
                                                    test_size=0.04,
                                                    stratify=session_df.label.values, random_state=6)

    # Retrieve the images from sessions in each subset
    train_df = df_.loc[df_.sessionID.isin(train_sessions)]
    test_df = df_.loc[df_.sessionID.isin(test_sessions)]


    train_session_df = train_df[['sessionID', 'label']].groupby('sessionID').agg('max')

    tr_sessions, val_sessions, _, _ = train_test_split(train_session_df.index.values,
                                                    train_session_df.label.values,
                                                    test_size=0.2,
                                                    stratify=train_session_df.label.values, random_state=6)

    tr_df = train_df.loc[df_.sessionID.isin(tr_sessions)]
    val_df = train_df.loc[df_.sessionID.isin(val_sessions)]    

    print(len(tr_df), len(val_df), len(test_df))
    return tr_df, val_df, test_df, weights

def load_imagenet16(folder, weakly=False, cropped=True, class_weights=None):
    classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    le = LabelEncoder()
    le.fit(classes)
    img_path = os.path.join(folder, 'DATASETS/16-class-ImageNet')
    if(weakly):
        mask_path = os.path.join(folder, 'DATASETS/16-class-ImageNet/masks')
    
    df = pd.read_csv(os.path.join(img_path, 'data_balanced.csv'))
    if(cropped):
        img_path = os.path.join(img_path, 'cropped')
    else:
        img_path = os.path.join(img_path, 'images')
    
    if(weakly):
        df['maskID'] = [os.path.join(mask_path, df.iloc[datum]['label'], df.iloc[datum]['imageID'][:-5] + '_mask.JPEG') for datum in df.index.values]
    df['imageID'] = [os.path.join(img_path, df.iloc[datum]['label'], df.iloc[datum]['imageID']) for datum in df.index.values]

    df['label'] = le.fit_transform(df['label'])

    weights = compute_class_weight(class_weights, np.unique(df.label.values), df.label.values)

    tr_sessions, test_sessions, _, _ = train_test_split(df.index.values,
                                                    df.label.values,
                                                    test_size=0.02,
                                                    stratify=df.label.values, random_state=6)

    # Retrieve the images from sessions in each subset
    train_df = df.loc[df.index.isin(tr_sessions)]
    test_df = df.loc[df.index.isin(test_sessions)]

    tr_sessions, val_sessions, _, _ = train_test_split(train_df.index.values,
                                                    train_df.label.values,
                                                    test_size=0.1,
                                                    stratify=train_df.label.values, random_state=6)

    tr_df = train_df.loc[train_df.index.isin(tr_sessions)]
    val_df = train_df.loc[train_df.index.isin(val_sessions)]    

    print(len(tr_df), len(val_df), len(test_df))

    return tr_df, val_df, test_df, weights 

def load_data(folder, dataset='simplified_no_colour', test=False, weakly=False, cropped=True, class_weights=None):
    synt = ['simplified_no_colour']
    if(dataset in synt):
        return load_synthetic_dataset(folder=folder, dataset=dataset, test=test, weakly=weakly, class_weights=class_weights)
    elif(dataset == 'NIH-NCI'):
        return load_NIH_NCI(folder=folder, weakly=weakly, cropped=cropped, class_weights=class_weights)
    elif(dataset == 'imagenet16'):
        return load_imagenet16(folder=folder, weakly=weakly, cropped=cropped, class_weights=class_weights)
        
if __name__ == "__main__":
    load_imagenet16('/media/TOSHIBA6T/ICRTO')