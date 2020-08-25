import tensorflow as tf
import numpy as np
import cv2
import os
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from albumentations import (
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    RandomBrightnessContrast,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    Flip,
    OneOf,
    Compose,
)


def strong_aug(p=0.5):
    """ Data augmentation function

    Keyword Arguments:
        p {float} -- probability of applying data augmentation (default: {.5})

    Returns:
        [Compose] -- a set of data augmentation operations
    """
    return Compose(
        [
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise(),], p=0.2),
            OneOf(
                [
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
            ),
            OneOf(
                [
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
            HueSaturationValue(p=0.3),
        ],
        p=p,
    )


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=32,
        img_size=(224, 224),
        num_classes=2,
        preprocess=utils.norm,
        masks=False,
        aug_prob=0,
        shuffle=True,
    ):
        super(DataGenerator, self).__init__()
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.preprocess = preprocess
        self.masks = masks
        self.aug_prob = aug_prob
        self.shuffle = shuffle

        self.batch_labels = []
        self.batch_names = []
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        self.batch_labels = []
        self.batch_names = []

        batch_index = self.idxs[index * self.batch_size : (index + 1) * self.batch_size]

        img_names = self.df.iloc[batch_index]["imageID"].values
        imgs = [cv2.imread(img_name, cv2.IMREAD_UNCHANGED) for img_name in img_names]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        imgs = [
            cv2.resize(img, self.img_size) for img in imgs if img.shape != self.img_size
        ]

        masks = np.zeros_like(imgs)
        if self.masks:
            mask_names = self.df.iloc[batch_index]["maskID"].values
            masks = [
                cv2.imread(mask_name, cv2.IMREAD_UNCHANGED) for mask_name in mask_names
            ]
            masks = [cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) for mask in masks]

            masks = [
                cv2.resize(mask, self.img_size)
                for mask in masks
                if mask.shape != self.img_size
            ]

        # data augmentation (if aug_prob > 0)
        aug = strong_aug(p=self.aug_prob)
        if self.masks:
            augmented_instances = [
                aug(image=img, mask=mask) for (img, mask) in zip(imgs, masks)
            ]
            imgs = [augmented["image"] for augmented in augmented_instances]
            mask = [augmented["mask"] for augmented in augmented_instances]
        else:
            augmented_instances = [aug(image=img) for img in imgs]
            imgs = [augmented["image"] for augmented in augmented_instances]

        # preprocessing
        imgs = [self.preprocess(img) for img in imgs]
        imgs = np.asarray(imgs)

        if self.masks:
            mask = [self.preprocess(mask) for mask in masks]
            masks = np.asarray(masks)

        labels = self.df.iloc[batch_index]["label"].values
        labels = list(map(int, labels))
        y = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

        self.batch_labels.extend(labels)
        self.batch_names.extend([img_name.split("/")[-1] for img_name in img_names])
        return (
            imgs,
            {"classifier": y, "explainer": masks},
        )

    def on_epoch_end(self):
        self.idxs = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


def load_synthetic_dataset(folder, masks=False, class_weights=None):
    """ Loads synthetic dataset

    Arguments:
        folder {str} -- directory where dataset is stored

    Keyword Arguments:
        dataset {str} -- selects which synthetic dataset to load (default: {'simplified_no_colour'})
        masks {bool} -- whether to load object detection masks (only need for hybrid loss) (default: {False})
        class_weights {str} -- whether to use class weights (default: {None})

    Returns:
        [misc] -- training, validation and testing dataframes, as well as class weights and class names
    """
    # encode class names
    classes = ["neg", "pos"]

    dtset_folder = os.path.join(folder, "trainval")
    test_dtset_folder = os.path.join(folder, "test")
    test_df = pd.read_excel(
        os.path.join(test_dtset_folder, "data.xlsx"), index_col=None
    )
    test_df["imageID"] = [
        os.path.join(test_dtset_folder, datum) for datum in test_df.imageID.values
    ]
    if masks:  # also load object detection masks
        test_df["maskID"] = [
            datum[:-4] + "_mask.jpg" for datum in test_df.imageID.values
        ]

    df = pd.read_excel(os.path.join(dtset_folder, "data.xlsx"), index_col=None)
    df["imageID"] = [os.path.join(dtset_folder, datum) for datum in df.imageID.values]

    if masks:  # also load object detection masks
        df["maskID"] = [datum[:-4] + "_mask.jpg" for datum in df.imageID.values]

    weights = compute_class_weight(
        class_weights, classes=np.unique(df.label.values), y=df.label.values
    )

    # stratified partitioning by ground-truth (train-test)
    tr_sessions, val_sessions = train_test_split(
        df.index.values, stratify=df.label.values, random_state=6
    )
    tr_df = df.loc[df.index.isin(tr_sessions)]
    val_df = df.loc[df.index.isin(val_sessions)]

    print(len(tr_df), len(val_df), len(test_df))
    return tr_df, val_df, test_df, dict(enumerate(weights)), classes


def load_NIH_NCI(folder, masks=False, class_weights=None):
    """ Loads NIH-NCI cervical cancer dataset

    Arguments:
        folder {str} -- directory where dataset is stored

    Keyword Arguments:
        masks {bool} -- whether to load object detection masks (only need for hybrid loss) (default: {False})
        class_weights {str} -- whether to use class weights (default: {None})

    Returns:
        [misc] -- training, validation and testing dataframes, as well as class weights and class names
    """
    # encode class names
    classes = ["healthy", "cancer"]
    img_path = folder
    dft = pd.DataFrame()
    for subpath in ["ALTS", "Biopsy", "CVT", "NHS"]:
        df = pd.read_excel(
            os.path.join(img_path, "data", subpath, "covariate_data.xls"), header=4
        )

        for ft, missing in [
            ("WRST_HIST_AFTER_DT", "."),
            ("HPV_DT", "."),
            ("AGE_GRP", "-1"),
            ("HPV_STATUS", "-1"),
            ("WRST_HIST_AFTER", -2),
        ]:
            df.loc[df[ft].astype(str) == missing, ft] = np.nan
            df[ft] = df[ft].astype(np.float)

        for i in np.arange(df.shape[0]):
            row = df.iloc[i]
            df.loc[i, "path"] = os.path.join(img_path, "data", subpath, row.GG_IMAGE_ID)

        dft = pd.concat([dft, df])

    dft = dft.dropna(subset=["WRST_HIST_AFTER"])
    df_ = pd.DataFrame(
        {"label": [0 if (int(hist) <= 1) else 1 for hist in dft.WRST_HIST_AFTER]}
    )

    df_["imageID"] = [p[:-4] + ".jpg" for p in dft.path]

    if masks:  # also load object detection masks
        df_["maskID"] = [p[:-4] + "_mask.jpg" for p in dft.path]
    df_["sessionID"] = [index for index, _ in enumerate(dft.GG_PATIENT_ID)]

    weights = compute_class_weight(
        class_weights, classes=np.unique(df.label.values), y=df.label.values
    )

    session_df = df_[["sessionID", "label"]].groupby("sessionID").agg("max")

    # stratified partitioning by ground-truth (train-test)
    train_sessions, test_sessions, _, _ = train_test_split(
        session_df.index.values,
        session_df.label.values,
        test_size=0.05,
        stratify=session_df.label.values,
        random_state=6,
    )

    # retrieve the images from sessions in each subset
    train_df = df_.loc[df_.sessionID.isin(train_sessions)]
    test_df = df_.loc[df_.sessionID.isin(test_sessions)]
    train_session_df = train_df[["sessionID", "label"]].groupby("sessionID").agg("max")

    # stratified partitioning by ground-truth (train-val)
    tr_sessions, val_sessions, _, _ = train_test_split(
        train_session_df.index.values,
        train_session_df.label.values,
        test_size=0.2,
        stratify=train_session_df.label.values,
        random_state=6,
    )

    tr_df = train_df.loc[df_.sessionID.isin(tr_sessions)]
    val_df = train_df.loc[df_.sessionID.isin(val_sessions)]

    print(len(tr_df), len(val_df), len(test_df))
    return tr_df, val_df, test_df, dict(enumerate(weights)), classes


def load_imagenetHVZ(folder, masks=False, class_weights=None):
    """ Loads imagenetHVZ dataset

    Arguments:
        folder {str} -- directory where dataset is stored

    Keyword Arguments:
        masks {bool} -- whether to load object detection masks (only need for hybrid loss) (default: {False})
        class_weights {str} -- whether to use class weights (default: {None})

    Returns:
        [misc] -- training, validation and testing dataframes, as well as class weights and class names
    """
    # encode class names
    classes = ["horse", "zebra"]
    le = LabelEncoder()
    le.fit(classes)

    img_path = folder

    df = pd.read_csv(os.path.join(img_path, "data.csv"))

    img_path = os.path.join(img_path, "images")

    if masks:  # also load object detection masks
        mask_path = os.path.join(folder, "masks")
        df["maskID"] = [
            os.path.join(
                mask_path,
                df.iloc[datum]["label"],
                df.iloc[datum]["imageID"][:-5] + "_mask.JPEG",
            )
            for datum in df.index.values
        ]
    df["imageID"] = [
        os.path.join(img_path, df.iloc[datum]["label"], df.iloc[datum]["imageID"])
        for datum in df.index.values
    ]

    df["label"] = le.fit_transform(df["label"])

    weights = compute_class_weight(
        class_weights, classes=np.unique(df.label.values), y=df.label.values
    )

    tr_sessions, test_sessions, _, _ = train_test_split(
        df.index.values,
        df.label.values,
        test_size=0.149,
        stratify=df.label.values,
        random_state=6,
    )

    # Retrieve the images from sessions in each subset
    train_df = df.loc[df.index.isin(tr_sessions)]
    test_df = df.loc[df.index.isin(test_sessions)]

    tr_sessions, val_sessions, _, _ = train_test_split(
        train_df.index.values,
        train_df.label.values,
        test_size=0.2,
        stratify=train_df.label.values,
        random_state=6,
    )

    tr_df = train_df.loc[train_df.index.isin(tr_sessions)]
    val_df = train_df.loc[train_df.index.isin(val_sessions)]

    print(len(tr_df), len(val_df), len(test_df))

    return tr_df, val_df, test_df, dict(enumerate(weights)), classes


def load_data(folder, dataset="imagenetHVZ", masks=False, class_weights=None):
    """ Triage function to select corresponding loading function according to chosen dataset

    Arguments:
        folder {str} -- directory where dataset is stored

    Keyword Arguments:
        dataset {str} -- dataset to choose (default: {'imagenetHVZ'})
        masks {bool} -- whether to load object detection masks (only needed for hybrid loss) (default: {False})
        class_weights {str} -- whether to use class weights (default: {None})

    Returns:
        [misc] -- training, validation and testing dataframes, as well as class weights and class names
    """
    if dataset == "synthetic":
        return load_synthetic_dataset(
            folder=folder, masks=masks, class_weights=class_weights
        )
    elif dataset == "NIH-NCI":
        return load_NIH_NCI(folder=folder, masks=masks, class_weights=class_weights)
    elif dataset == "imagenetHVZ":
        return load_imagenetHVZ(folder=folder, masks=masks, class_weights=class_weights)
