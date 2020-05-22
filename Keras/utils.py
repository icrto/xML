import numpy as np
np.random.seed(42)
import sys
import os
import cv2
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_colwidth = 1000
from sklearn.model_selection import train_test_split
from keras.callbacks import History, LearningRateScheduler, ReduceLROnPlateau, Callback
import datetime
import csv
from random import shuffle
from ModelCheckpoint import ModelCheckpoint
from keras import optimizers
from keras import backend as K
from keras import Model
from ModelEncoder import ModelEncoder

def unsupervised_explanation_loss(beta):
    def loss(y_true, y_pred):
        reg = 0.
        reg += beta*K.mean(K.abs(y_pred)) + (1-beta)*(K.mean(K.abs(y_pred[: -1] - y_pred[1:])) + K.mean(K.abs(y_pred[:, : -1] - y_pred[:, 1:])))
        return reg
    return loss

def weaklysupervised_explanation_loss(y_true, y_pred):
    return np.abs(np.divide(np.sum(np.multiply(1-y_true, y_pred)+K.epsilon()), np.sum(1-y_true)+K.epsilon()))

def hybrid_explanation_loss(beta1, beta2):
    def hybridloss(y_true, y_pred):
        beta3 = 1.0-beta1-beta2
        total = beta1+beta2+beta3
        l = ((beta1/total)*K.mean(K.abs(y_pred)) + 
        (beta2/total)*(K.mean(K.abs(y_pred[: -1] - y_pred[1:])) + K.mean(K.abs(y_pred[:, : -1] - y_pred[:, 1:]))) +
        (beta3/total)*np.abs(np.divide(np.sum(np.multiply(1-y_true, y_pred)), np.sum(1-y_true))))
        return l
    return hybridloss    

def active_pixels(y_, p_):
    return K.mean(p_ > 0) * 100

def preprocess(img):
    dst = img / 256.
    return dst

class LRCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print(lr) 

def image_generator(df, preprocess_function, image_column='imageID', mask_column='maskID', gt_column='GT_real', batch_size=32, img_size=(224, 224),
                    num_classes=2, loss='unsup'):
    while True:
        index = np.arange(df.shape[0])
        np.random.shuffle(index)
        
        for bid in range(len(index) // batch_size):
            batch_index = index[bid * batch_size: (bid + 1) * batch_size]
            imgs = df.iloc[batch_index][image_column].values
            if(loss == 'weakly' or loss == 'hybrid'):
                masks = df.iloc[batch_index][mask_column].values

            
            #for j in range(len(imgs)):
              #print(imgs[j])
              #imagem = cv2.imread(imgs[j])
              #imagem = cv2.resize(imagem, img_size)
            
            imgs = [cv2.imread(i) for i in imgs]
            imgs = [cv2.resize(i, img_size) for i in imgs]
            imgs = [preprocess_function(i) for i in imgs]
            imgs = np.asarray(imgs)

            if(loss == 'weakly' or loss == 'hybrid'):
                masks = [cv2.imread(j) for j in masks]
                masks = [cv2.resize(j, img_size) for j in masks]
                masks = [preprocess(j) for j in masks]
                masks = np.asarray(masks)

            labels = df.iloc[batch_index][gt_column].values
            labels = list(map(int, labels))

            y = np.zeros((imgs.shape[0], num_classes))
            y[np.arange(imgs.shape[0]), labels] = 1 #one hot encoding
           
            if(loss == 'weakly' or loss == 'hybrid'):
                yield imgs, {'decision-maker': y, 'explainer': masks}
            else:
                yield imgs, {'decision-maker': y, 'explainer': np.zeros_like(imgs)}

 

def optimizer(opt='adadelta', lr=1e-4, decay=1e-4, momentum=0.9):
    opt = opt.lower()
    if(opt == 'sgd'):
        return optimizers.SGD(lr=lr, decay=decay, momentum=momentum)
    elif(opt == 'adadelta'):
        return optimizers.Adadelta()
    else:
            print("Invalid Optimizer, please choose another.")
            sys.exit(-1)

def create_folder(folder, clf, dtset, opt_str, lr, decay, batch_size, epochs, loss, alfa, beta, beta1, beta2):
    print(folder)
    if(not os.path.exists(folder)):
        os.mkdir(folder)
    else:
        print('Folder already exists.')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(folder, timestamp)
    epochs = epochs.replace(",", "_")
    if(opt_str == 'adadelta'):
        lr = 'default'
        decay = 'default'
    if(loss == 'hybrid'):
        results_path = (results_path + '_' + clf + '_' + dtset + '_' + loss +
            '_optimizer_'  + opt_str + 
            '_lr_' + str(lr) + '_decay_' + str(decay) + 
            '_batch_size_' + str(batch_size) + 
            '_epochs_' + epochs + '_alfa_' + str(alfa) + 
            '_beta1_' + str(beta1) + '_beta2_' + str(beta2))
    elif(loss == 'unsup'):
        results_path = (results_path + '_' + clf + '_' + dtset + '_' + loss +
                    '_optimizer_'  + opt_str + 
                    '_lr_' + str(lr) + '_decay_' + str(decay) + 
                    '_batch_size_' + str(batch_size) + 
                    '_epochs_' + epochs + '_alfa_'+ str(alfa) + 
                    '_beta_' + str(beta))
    elif(loss == 'weakly'):
        results_path = (results_path + '_' + clf + '_' + dtset + '_' + loss +
                    '_optimizer_'  + opt_str + 
                    '_lr_' + str(lr) + '_decay_' + str(decay) + 
                    '_batch_size_' + str(batch_size) + 
                    '_epochs_' + epochs + '_alfa_' + str(alfa))

    if(not os.path.exists(results_path)):   
        os.mkdir(results_path)
    else:
        print('Folder already exists.')
        sys.exit(-1)

    return timestamp, results_path

def plot_metric_train_val(nr_epochs, history, metric, path, filename, plot_title):
    x_values = np.linspace(1, nr_epochs, nr_epochs)
    plt.plot(x_values, history.history[metric], '--')
    plt.plot(x_values, history.history[str('val_' + metric)], ':')
    plt.title(plot_title)
    if('loss' in metric):
        plt.ylabel('Loss')
    elif('acc' in metric):
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1.1])
    plt.xlabel('Epoch')
    plt.xlim([0, nr_epochs])
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(path, filename))
    plt.close()

def config_callbacks(model, factor, patience, min_lr, decmaker_model_path, exp_model_path, e2e_model_path):
    callbacks = []
    history = History()
    lrCallback = LRCallback()
    checkpointer = ModelCheckpoint(base_model=model.e2e_model, decmaker=model.decmaker.model, 
                                           explainer=model.explainer.model, filepath=e2e_model_path, 
                                           decmaker_filepath=decmaker_model_path, exp_filepath=exp_model_path,
                                           save_best_only=True,
                                           save_weights_only=False, verbose=True)
    callbacks.extend((history, lrCallback, checkpointer))
    if(model.clf == 'ResNet50Mod'):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr, verbose=1)
        callbacks.append(reduce_lr)
    return callbacks

def get_images_and_labels(img_size, df, preprocess_function, weakly):
    np.random.seed(42)
    
    idxs = np.arange(df.shape[0])

    np.random.shuffle(idxs)
    
    #idxs = idxs[:100]

    imgs = []
    imgs_preprocessed = []
    labels = []
    if(weakly):
        masks = []
    for idx in idxs:
        img = cv2.imread(df.imageID.values[idx])
        img = cv2.resize(img, img_size)
        imgs.append(img)
        img_aux = preprocess_function(img)
        imgs_preprocessed.append(img_aux)
        labels.append(int(df.GT_real.values[idx]))
        if(weakly):
            mask = cv2.imread(df.maskID.values[idx])
            mask = cv2.resize(mask, img_size)
            mask_aux = preprocess_function(mask)
            masks.append(mask_aux)
    if(weakly):
        return imgs, imgs_preprocessed, idxs, labels, masks
    else:
        return imgs, imgs_preprocessed, idxs, labels

def save_explanations(model, steps, bs, img_size, dtset, df, imgs, imgs_preprocessed, idxs, path, timestamp, phase, option=None, layer=None):
    print('Saving images...')
    idx = 0
    fixed_explanation_input =  np.ones(tuple([bs] + list(img_size) + [1]))

    for img_ in imgs_preprocessed:
        img_name =  df.imageID.values[idxs[idx]].split('/')[-1]
        if((dtset == 'test') and (phase == 0 or phase == 1)):
            expl = model.predict([np.asarray([img_]), fixed_explanation_input], steps=steps)[0][0]
        else:
            expl = model.predict(np.asarray([img_]), steps=steps)[0][0]

        plt.figure(1)
        plt.axis('off')
        plt.imshow((expl)[:, :, 0])
        if((option is not None) and (layer is not None)):
            #'hello {} world'.format(my_var)
            #'{}_phase{}_image_{}_{}_layer_{}_{}'.format(timestamp, str(phase), dtset, option, str(layer), df.imageID.values[idxs[idx]].split('/')[-1])
            plt.savefig(os.path.join(path, '{}_phase{}_image_{}_{}_layer_{}_{}'.format(timestamp, str(phase), dtset, option, str(layer), img_name)), bbox_inches='tight', transparent=True, pad_inches=0)
        else:
            plt.savefig(os.path.join(path, '{}_phase{}_image_{}_{}'.format(timestamp, str(phase), dtset, img_name)), bbox_inches='tight', transparent=True, pad_inches=0)

        i = 2
        expl_thr = []
        for thr in [0.25, 0.5, 0.75]:            
            expl_thr.append((expl > thr)[:, :, 0])
            if(thr == 0.25 or thr == 0.5): continue

            plt.figure(i, figsize=(15, 5))
            plt.subplot(131)
            ax = plt.gca()
            ax.relim()
            ax.autoscale()
            plt.title('Original Image')                                    
            plt.imshow(imgs[idx][:, :, ::-1])

            
            plt.subplot(132)
            plt.title('Explanation')
            plt.imshow((expl)[:, :, 0])

            plt.subplot(133)
            plt.imshow((expl > thr)[:, :, 0])
            plt.title('Explanation > ' + str(thr))
            if((option is not None) and (layer is not None)):
                plt.savefig(os.path.join(path, '{}_phase{}_ex_img_thr{}_{}_{}_layer_{}_{}'.format(timestamp, str(phase), str(thr), dtset, option, str(layer), img_name))) 
            else:
                plt.savefig(os.path.join(path, '{}_phase{}_ex_img_thr{}_{}_{}'.format(timestamp, str(phase), str(thr), dtset, img_name))) 
            
            plt.figure(i+1)
            plt.axis('off')
            plt.imshow((expl > thr)[:, :, 0])
            if((option is not None) and (layer is not None)):
                plt.savefig(os.path.join(path, '{}_phase{}_image_thr{}_{}_{}_layer_{}_{}'.format(timestamp, str(phase), str(thr), dtset, option, str(layer), img_name)), bbox_inches='tight', transparent=True, pad_inches=0)
            else:
                plt.savefig(os.path.join(path, '{}_phase{}_image_thr{}_{}_{}'.format(timestamp, str(phase), str(thr), dtset, img_name)), bbox_inches='tight', transparent=True, pad_inches=0)
            i += 2
        
        plt.figure(i, figsize=(15, 5))
        plt.subplot(151)
        ax = plt.gca()
        ax.relim()
        ax.autoscale()
        plt.title('Original Image')                                    
        plt.imshow(imgs[idx][:, :, ::-1])

        plt.subplot(152)
        plt.title('Explanation')
        plt.imshow((expl)[:, :, 0])

        plt.subplot(153)
        plt.imshow(expl_thr[0])
        plt.title('Explanation > 0.25')

        plt.subplot(154)
        plt.imshow(expl_thr[1])
        plt.title('Explanation > 0.5')

        plt.subplot(155)
        plt.imshow(expl_thr[2])
        plt.title('Explanation > 0.75')
        if((option is not None) and (layer is not None)):
            plt.savefig(os.path.join(path, '{}_phase{}_ex_img_all_thrs_{}_{}_layer_{}_{}'.format(timestamp, str(phase), dtset, option, str(layer), img_name))) 
        else:
            plt.savefig(os.path.join(path, '{}_phase{}_ex_img_all_thrs_{}_{}'.format(timestamp, str(phase), dtset, img_name))) 

        plt.close()
        #print(expl.min(), expl.max(), expl.mean())
        
        idx += 1

def class_predictions(model, steps, bs, img_size, dtset, df, imgs, idxs, labels, path, timestamp, phase, option=None, layer=None):

    if((option is not None) and (layer is not None)):
        file_name = os.path.join(path, timestamp + '_phase' + str(phase) + '_' + dtset + '_' + option + '_layer_' + str(layer) + '_predictions.csv')
    else:
        file_name = os.path.join(path, timestamp + '_phase' + str(phase) + '_' + dtset + '_predictions.csv')

    try:
        f = open(file_name,'w')
    except Exception as e:
        print(e)
        sys.exit(-1)

    f.write('ImageID, Label, Probabilities, Predicted\n')
    i = 0               
    for img in imgs: 
        #expl = model.explainer.model.predict(np.asarray([img]))[0]
        img_aux_a = np.asarray(img)
        #img_predict = img_aux_a.reshape(1, img_aux_a.shape[0], img_aux_a.shape[1], img_aux_a.shape[2])
        #expl_a = np.asarray(expl)

        #exp_predict = expl_a.reshape(1, expl_a.shape[0], expl_a.shape[1], expl_a.shape[2])
        #clss = model.decmaker.model.predict([exp_predict, img_predict], batch_size=1)

        if(dtset == 'test' and (phase == 0 or phase == 1)):
            fixed_explanation_input =  np.ones(tuple([bs] + list(img_size) + [1]))
            clss = model.predict([np.asarray([img]), fixed_explanation_input], steps=steps)[1][0]
        else:
            clss = model.predict(np.asarray([img]), steps=steps)[1][0]

        dec = clss.argmax(axis=-1)
        label = labels[i]

        f.write(str(df.imageID.values[idxs[i]].split('/')[-1]) + ',' + str(label) + ',' + str(clss) + ',' + str(dec) + '\n')
        i+= 1
    f.close()

def compute_accuracy(model, num_classes, dtset, imgs, labels, path, timestamp, img_size, phase, option=None, layer=None, masks=None):
    y = np.zeros((np.asarray(imgs).shape[0], num_classes))
    y[np.arange(np.asarray(imgs).shape[0]), labels] = 1 #one hot encoding
    if((dtset == 'test') and (phase == 0 or phase == 1)):
        fixed_explanation_input =  np.ones(tuple([np.asarray(imgs).shape[0]] + list(img_size) + [1]))
        x = [np.asarray(imgs), fixed_explanation_input]
    else:
        x = np.asarray(imgs)
    #acc = self.decmaker.evaluate(x={'decision-explanation-input': np.asarray(expls), 'decision-image-input': np.asarray(imgs)}, y=y, verbose=1)
    if(masks is not None):
        acc = model.evaluate(x=x, y={'decision-maker': y, 'explainer': np.asarray(masks)}, verbose=1, batch_size=1)
    else:
        acc = model.evaluate(x=x, y={'decision-maker': y, 'explainer': np.zeros_like(imgs)}, verbose=1, batch_size=1)

    print(acc)
    print('Loss:', acc[2])
    print('Accuracy:', acc[4])

    if((option is not None) and (layer is not None)):
        file_name = os.path.join(path, timestamp + '_acc_' + dtset + '_' + option + '_layer_' + str(layer) + '.txt')
    else:
        file_name = os.path.join(path, timestamp + '_acc_' + dtset + '.txt')
    try:
        file = open(file_name, 'w') 
    except Exception as e:
        print(e)
        sys.exit(-1)
    file.write('Loss: ' + str(acc[2]))
    file.write('Acc: ' + str(acc[4]))
    file.close()

def save_history(history, path, filename):
    hist_df = pd.DataFrame.from_dict(history.history, orient='index')
    hist_path = os.path.join(path, filename)
            
    try:
        #create file
        f = open(hist_path, "x")
    except:
        print('Could not create history file.')
        sys.exit(-1)
    
    with open(hist_path, 'wb+') as file_pi:
        pickle.dump(hist_df, file_pi)

def save_model(model, path):
    json_file = os.path.join(path, 'model.json')
    print("Saving final model.")
    with open(json_file, 'w') as f:
        json.dump(ModelEncoder().encode(model), f)

def load_synthetic(path, dtset, weakly=False, random_labels=False):
    img_path = os.path.join(path, dtset)
    data_file = os.path.join(path, 'data_' + dtset + '.txt')
    test_img_path = os.path.join(path, dtset + '_test')
    test_file = os.path.join(path, 'data_' + dtset + '_test.txt')
    with open(data_file, "rb") as fp:   # load data
            data = pickle.load(fp)

    if(random_labels):
        np.random.seed(42)
        labels = [str(1) if int(datum['exists']) > 0 else str(0) for datum in data]
        labels_randomized = np.random.permutation(labels)
        df = pd.DataFrame({'GT_real': labels_randomized})
    else:
        df = pd.DataFrame({'GT_real': [str(1) if int(datum['exists']) > 0 else str(0) for datum in data]})

    df['imageID'] = [os.path.join(img_path, datum['filename']) for datum in data]
    if(weakly):
        df['maskID'] = [os.path.join(img_path, datum['filename'][:-4] + '_mask.jpg') for datum in data]
    df['sessionID'] = [index for index, _ in enumerate(data)]

    session_df = df[['sessionID', 'GT_real']].groupby('sessionID').agg('max')

    # Stratified partitioning by ground-truth
    tr_sessions, val_sessions, _, _ = train_test_split(session_df.index.values,
                                                        session_df.GT_real.values,
                                                        stratify=session_df.GT_real.values)

    # Retrieve the images from sessions in each subset
    tr_df = df.loc[df.sessionID.isin(tr_sessions)]
    val_df = df.loc[df.sessionID.isin(val_sessions)]

    with open(test_file, "rb") as fp:   # load data
        test_data = pickle.load(fp)

    if(random_labels):
        np.random.seed(42)
        test_labels = [str(1) if int(datum['exists']) > 0 else str(0) for datum in test_data]
        test_labels_randomized = np.random.permutation(test_labels)
        test_df = pd.DataFrame({'GT_real': test_labels_randomized})
    else:
        test_df = pd.DataFrame({'GT_real': [str(1) if int(datum['exists']) > 0 else str(0) for datum in test_data]})

    test_df['imageID'] = [os.path.join(test_img_path, datum['filename']) for datum in test_data]
    å
    if(weakly):
        test_df['maskID'] = [os.path.join(test_img_path, datum['filename'][:-4] + '_mask.jpg') for datum in test_data]
    test_df['sessionID'] = [index for index, _ in enumerate(test_data)]

    print(len(tr_df), len(val_df), len(test_df))
    return tr_df, val_df, test_df

def load_data(dtset, path='/media/TOSHIBA6T/ICRTO', weakly=False, preprocessed=True, random_labels=False):
    flag = 1
    classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    synt = ['triangles_vs_stars_simplified', 'simplified_no_colour', 'simplified', 'normal', 'med', 'multiple_targets_no_colour', 
    'triangles_vs_squares', 'triangles_vs_stars', 'noseg_nooverlap_nocircles', 'noseg_nooverlap', 'noseg']

    if(dtset in synt):
        return load_synthetic(os.path.join(path, 'Synthetic Dataset'), dtset, weakly, random_labels)
    elif(dtset in '16classImageNetBalancedTINY'):
        directory = os.path.join(path, 'DATASETS/16-class-ImageNet/data/train-val-test')
        tr_df = pd.read_pickle(os.path.join(directory, 'train_balanced_tiny_df.txt'))
        val_df = pd.read_pickle(os.path.join(directory, 'val_balanced_tiny_df.txt'))
        test_df = pd.read_pickle(os.path.join(directory, 'test_balanced_tiny_df.txt'))
    elif('NIH-NCI' in dtset):
        file_ = '/media/icrto/TOSHIBA6T/ICRTO/DATASETS/NIH-NCI'
        img_path = "/media/icrto/TOSHIBA6T/ICRTO/DATASETS/NIH-NCI"
        dft = pd.DataFrame()
        for subpath in ['ALTS', 'Biopsy', 'CVT', 'NHS']:
            df = pd.read_excel(os.path.join(file_, 'data', subpath, 'covariate_data.xls'), header=4)
            
            for ft, missing in [('WRST_HIST_AFTER_DT', '.'), ('HPV_DT', '.'),
                                ('AGE_GRP', '-1'), ('HPV_STATUS', '-1'),
                                ('WRST_HIST_AFTER', -2)]:
                df.loc[df[ft].astype(str) == missing, ft] = np.nan
                df[ft] = df[ft].astype(np.float)
            
            for i in np.arange(df.shape[0]):   
                row = df.iloc[i]  
                if(weakly or not preprocessed): 
                    df.loc[i, 'path'] =  os.path.join(img_path, 'data2', subpath, row.GG_IMAGE_ID)              
                elif(preprocessed):
                    df.loc[i, 'path'] =  os.path.join(img_path, 'preprocessed', subpath, row.GG_IMAGE_ID)

            dft = pd.concat([dft, df])
                  
        dft = dft.dropna(subset=['WRST_HIST_AFTER'])
        if(random_labels):
            np.random.seed(42)
            labels = [str(0) if(int(hist) <= 1) else str(1) for hist in dft.WRST_HIST_AFTER]
            labels_randomized = np.random.permutation(labels)
            df_ = pd.DataFrame({'GT_real': labels_randomized})
        else:
            df_ = pd.DataFrame({'GT_real': [str(0) if(int(hist) <= 1) else str(1) for hist in dft.WRST_HIST_AFTER]})
        df_['imageID'] = [p for p in dft.path]
        if(weakly):
            df_['maskID'] = [p[:-4] + '_mask.jpg' for p in dft.path]
        df_['sessionID'] = [index for index, _ in enumerate(dft.GG_PATIENT_ID)]
        session_df = df_[['sessionID', 'GT_real']].groupby('sessionID').agg('max')

        # Stratified partitioning by ground-truth
        train_sessions, test_sessions, _, _ = train_test_split(session_df.index.values,
                                                           session_df.GT_real.values,
                                                           test_size=0.2,
                                                           stratify=session_df.GT_real.values)

        # Retrieve the images from sessions in each subset
        train_df = df_.loc[df_.sessionID.isin(train_sessions)]
        test_df = df_.loc[df_.sessionID.isin(test_sessions)]


        train_session_df = train_df[['sessionID', 'GT_real']].groupby('sessionID').agg('max')

        tr_sessions, val_sessions, _, _ = train_test_split(train_session_df.index.values,
                                                           train_session_df.GT_real.values,
                                                           test_size=0.2,
                                                           stratify=train_session_df.GT_real.values)

        tr_df = train_df.loc[df_.sessionID.isin(tr_sessions)]
        val_df = train_df.loc[df_.sessionID.isin(val_sessions)]    
        
    else:
        print('Unknown dataset')
        sys.exit(-1)

    print(len(tr_df), len(val_df), len(test_df))
    return tr_df, val_df, test_df 

def save_untrained_models(base_model, decmaker, explainer, filepath, decmaker_filepath, exp_filepath):
    trainable_config = []
    print("Setting all layers to trainable")
    for layer in base_model.layers:
        #print(layer, layer.trainable)
        trainable_config.append(layer.trainable)
        layer.trainable = True
        if(isinstance(layer, Model)):
            for sublayer in layer.layers:
                #print(sublayer, sublayer.trainable)
                trainable_config.append(sublayer.trainable)
                sublayer.trainable = True
        #print(" ")

    base_model.save(filepath, overwrite=True)
    decmaker.save(decmaker_filepath, overwrite=True)
    explainer.save(exp_filepath, overwrite=True)
    #self.model.save(full_filepath, overwrite=True)

    print("Reverting previous operation")
    i = 0
    for layer in base_model.layers:
            #print("Before", layer, layer.trainable)
            layer.trainable = trainable_config[i]
            i += 1
            #print("After", layer, layer.trainable)
            if(isinstance(layer, Model)):
                for sublayer in layer.layers:
                    #print("Before", sublayer, sublayer.trainable)
                    sublayer.trainable = trainable_config[i]
                    i += 1
                    #print("After", sublayer, sublayer.trainable)
            #print(" ")
    #self.model.save(full_filepath, overwrite=True)
