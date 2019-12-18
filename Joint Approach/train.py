#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import sys
from ExplainerClassifierCNN import ExplainerClassifierCNN
import utils
import argparse
import numpy as np
np.random.seed(42)
import random as rn
rn.seed(1234)
import itertools as it
from tensorflow.python.client import device_lib
from keras import backend as K


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--gpu", type=str, default='1',
                    help="Which gpus to use in CUDA_VISIBLE_DEVICES.")
parser.add_argument("--nr_gpus", type=int, default=1,
                    help="Number of gpus to use.")                    
parser.add_argument("dataset", type=str,
                    help="Dataset to load.")
parser.add_argument("--dataset_path", type=str, default="/media/TOSHIBA6T/ICRTO",
                    help="Folder where dataset is located.")                  
parser.add_argument("--nr_classes", type=int, default=2,
                    help="Number of target classes.")
parser.add_argument("nr_epochs", type=str,
                    help="Number of epochs for each of the 3 training phases as an array, for example: 50,100,50.")
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Training batch size.")
parser.add_argument("alfas", type=str,
                    help="Loss = alfa * Lclassif + (1-alfa) * Lexplic")                   
parser.add_argument("--betas", type=str,
                    help="Lexplic_unsup = beta * L1 + (1-beta) * Total Variation")    
parser.add_argument("--betas1", type=str,
                    help="Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss")  
parser.add_argument("--betas2", type=str,
                    help="Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss")  
parser.add_argument("--nr_searches", type=int, default=1,
                    help="Number of random searches to perform.")   
parser.add_argument("folder", type=str, 
                    help="Directory where images and models are to be stored.")
parser.add_argument("--pretrained", default=False,
                    help="True if one wants to load pre trained models in training phase 1, False otherwise.")
parser.add_argument("--loss", type=str, default='weakly',
                    help="Specifiy if loss is weakly/hybrid/unsupervised or not.")
parser.add_argument("--opt", type=str, default='adadelta', 
                    help="Optimizer to use. Either adadelta or sgd.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Learning rate to use with sgd.")
parser.add_argument("--decay", type=float, default=0.0001,
                    help="Learning rate decay to use with sgd.")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                    help="Momentum to use with sgd.")
parser.add_argument("-min_lr", "--min_learning_rate", type=float, default=1e-5,
                    help="Minimum learning rate to use with sgd and with ReduceLearningRateonPlateau.")
parser.add_argument("--patience", type=int, default=10,
                    help="Patience (number of epochs for a model to be considered as converged) to use with sgd and with ReduceLearningRateonPlateau.")
parser.add_argument("--factor", type=float, default=0.2, 
                    help="Learning rate changing factor to use with sgd and with ReduceLearningRateonPlateau.")
parser.add_argument("--exp_conv_num", type=int, default=3,
                    help="Number of convolutional explainer stages.")
parser.add_argument("--exp_conv_conseq", type=int, default=2,
                    help="Number of convolutional layers in each explainer stage.")
parser.add_argument("--exp_conv_filters", type=int, default=32,
                    help="Number of filters for the initial convolutional explainer stage.")
parser.add_argument("--exp_conv_filter_size", type=tuple, default=(3,3),
                    help="Filter size for the initial explainer stage's filters.")
parser.add_argument("--exp_conv_activation", type=str, default='relu',
                    help="Activation function for the explainer's convolutional section.")
parser.add_argument("--exp_pool_size", type=int, default=2,
                    help="Explainer pooling layers' size.")
parser.add_argument("--dec_conv_num", type=int, default=4,
                    help="Number of transpose convolutional explainer stages.")   
parser.add_argument("--dec_conv_conseq", type=int, default=2,
                    help="Number of transpose convolutional layers in each explainer stage.")                    
parser.add_argument("--dec_conv_filters", type=int, default=32,
                    help="Number of filters for the initial transpose convolutional explainer stage.")
parser.add_argument("--dec_conv_filter_size", type=tuple, default=(3,3),
                    help="Filter size for the initial explainer transpose convolutional stage's filters.")
parser.add_argument("--dec_conv_activation", type=str, default='relu',
                    help="Activation function for the explainer's transpose convolutional section.")
parser.add_argument("--dec_pool_size", type=int, default=2,
                    help="Explainer pooling layers' size - transpose convolution.")
parser.add_argument("--dec_dense_num", type=int, default=2,
                    help="Number of dense layers in the explainer.")
parser.add_argument("--dec_dense_width", type=int, default=128,
                    help="Width of dense layers in the explainer.")
parser.add_argument("--dec_dense_activation", type=str, default='sigmoid',
                    help="Activation function for the explainer's dense layers.")
parser.add_argument("--img_size", type=tuple, default=(224, 224), help="Input image size.")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (VGG classifier).")
parser.add_argument("-clf", "--classifier", type=str, default='VGG', help="Classifier (VGG or ResNet50).")
parser.add_argument("--cropped", default=True, help="Load NIH-NCI dataset with cropped images if True, otherwise False.")
parser.add_argument("--random_labels", default=False, help="Randomize labels.")
parser.add_argument("--augment", default=False, help="Perform data augmentation.")

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# confirm TensorFlow sees the GPU
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
assert len(K.tensorflow_backend._get_available_gpus()) > 0


nr_gpus = args.nr_gpus

dtset = args.dataset
loss = args.loss
weakly = False
if(loss == 'weakly' or loss == 'hybrid'):
    weakly = True

tr_df, val_df, test_df = utils.load_data(dtset, path=args.dataset_path, weakly=weakly, preprocessed=args.cropped, random_labels=args.random_labels)
nr_classes = args.nr_classes

eps = args.nr_epochs.split(',')
nr_epochs = np.array([int(x) for x in eps])

steps_per_epoch = int(np.ceil(len(tr_df)/args.batch_size))

alfa = np.array([float(x) for x in args.alfas.split(',')])
hyperparams = {'alfas': alfa}
if(loss == 'unsup'):
    if(args.betas is None):
        print("Please define at least one value for beta.")
        sys.exit(-1)
    hyperparams['betas'] = np.array([float(x) for x in args.betas.split(',')])
elif(loss == 'weakly'):
    pass
elif(loss == 'hybrid'):
    if(args.betas1 is None):
        print("Please define at least one value for beta1.")
        sys.exit(-1)
    if(args.betas2 is None):
        print("Please define at least one value for beta2.")
        sys.exit(-1)
    hyperparams['betas1'] = np.array([float(x) for x in args.betas1.split(',')])
    hyperparams['betas2'] = np.array([float(x) for x in args.betas2.split(',')])
else:
    print('Invalid loss function. Try again with <unsup>, <weakly> or <hybrid>.')
    sys.exit(-1)

folder = args.folder

opt_str = args.opt
if(opt_str == 'adadelta'):
    opt = utils.optimizer(opt_str, None, None, None)
else:
    opt = utils.optimizer(opt_str, args.learning_rate, args.decay, args.momentum)

search = 0
print('Starting random search')

for item in rn.sample(list(it.product(*hyperparams.values())), k=args.nr_searches):
    search += 1
    print("Search %d of %d" % (search, args.nr_searches))

    a = item[0]
    print("alfa = %.2f" % a)

    if(loss == 'unsup'):
        b = item[1]
        loss_func = utils.unsupervised_explanation_loss(beta=b)
        print("beta = %.2f" % b)
        timestamp, path = utils.create_folder(folder, args.classifier, dtset, opt_str, args.learning_rate, args.decay, args.batch_size, args.nr_epochs, args.loss, a, b, None, None) 
    elif(loss == 'weakly'):
        loss_func = utils.weaklysupervised_explanation_loss
        timestamp, path = utils.create_folder(folder,  args.classifier, dtset, opt_str, args.learning_rate, args.decay, args.batch_size, args.nr_epochs, args.loss, a, None, None, None) 
    elif(loss == 'hybrid'):
        b1 = item[1]
        b2 = item[2]
        print("beta1 = %.2f" % b1)
        print("beta2 = %.2f" % b2)
        if(b1 + b2 >= 1.0):
            print('Ignored beta1 and beta2.')
            continue
        loss_func = utils.hybrid_explanation_loss(beta1=b1, beta2=b2)
        timestamp, path = utils.create_folder(folder,  args.classifier, dtset, opt_str, args.learning_rate, args.decay, args.batch_size, args.nr_epochs, args.loss, a, None, b1, b2) 

   

    model = ExplainerClassifierCNN(nr_gpus=nr_gpus, num_classes=nr_classes,
                 dec_conv_num=args.dec_conv_num, dec_conv_conseq=args.dec_conv_conseq, 
                 dec_conv_filters=args.dec_conv_filters,
                 dec_conv_filter_size=args.dec_conv_filter_size, dec_conv_activation=args.dec_conv_activation, 
                 dec_pool_size=args.dec_pool_size, dec_dense_num=args.dec_dense_num, dec_dense_width=args.dec_dense_width, 
                 dec_dense_activation=args.dec_dense_activation, img_size=args.img_size,
                 dropout=args.dropout, clf=args.classifier, loss=loss)


    if(weakly):
        val_imgs, val_imgs_preprocessed, val_idxs, val_labels, val_masks = utils.get_images_and_labels(model.img_size, test_df, model.decmaker.preprocess_function, weakly)
    else:
        val_imgs, val_imgs_preprocessed, val_idxs, val_labels = utils.get_images_and_labels(model.img_size, test_df, model.decmaker.preprocess_function, weakly)

    for phase in range(3):

        if(nr_epochs[phase] == 0): continue
       
        model.build_model(phase=phase, pretrained=args.pretrained)

        if(phase > 0):
            print('Trying to load weights from file: %s' % previous_model_path)
            try:
                model.e2e_model.load_weights(previous_model_path, by_name=False)
                print("Loaded e2e weights from " + previous_model_path)
            except Exception as e:
                print("Error loading pre-trained e2e model.")
                print(e)
                sys.exit(-1)

        if(phase == 0):
            model.explainer.freeze()
            print(model.explainer.model.layers[0].trainable)
        elif(phase == 1):
            model.decmaker.freeze()
            print(model.decmaker.model.layers[0].trainable)


        model.e2e_model_gpu.compile(optimizer=opt,
                                    loss_weights={'decision-maker': a,
                                                    'explainer': 1.0-a
                                                },
                                    loss={'explainer': loss_func,
                                            'decision-maker': 'categorical_crossentropy'
                                        },
                                    metrics={'decision-maker': ['accuracy'],
                                            'explainer': [utils.active_pixels]
                                            })
        model.e2e_model.summary()
        model.save_architecture(phase, timestamp, path)
        e2e_model_filename = timestamp + '_phase' + str(phase) + '_model.h5'
        e2e_model_path = os.path.join(path, e2e_model_filename)
        decmaker_model_filename = timestamp + '_phase' + str(phase) + '_decmaker_model.h5'
        decmaker_model_path = os.path.join(path, decmaker_model_filename)
        exp_model_filename = timestamp + '_phase' + str(phase) + '_exp_model.h5'
        exp_model_path = os.path.join(path, exp_model_filename)
        
        callbacks = utils.config_callbacks(model, args.factor, args.patience, args.min_learning_rate, decmaker_model_path, exp_model_path, e2e_model_path)
        model.history = callbacks[0]
        model.fit(tr_df, val_df, nr_epochs[phase], steps_per_epoch, callbacks, path, augment=args.augment, loss=loss)
        
        utils.plot_metric_train_val(nr_epochs[phase], model.history, 'decision-maker_loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_decmaker_loss.png'), 'Classifier Loss')
        utils.plot_metric_train_val(nr_epochs[phase], model.history, 'explainer_loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_explainer_loss.png'), 'Explainer Loss')
        utils.plot_metric_train_val(nr_epochs[phase], model.history, 'loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_global_loss.png'), 'Global Loss')
        utils.plot_metric_train_val(nr_epochs[phase], model.history, 'decision-maker_acc', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_decision-maker_acc.png'), 'Accuracy')

        utils.save_history(model.history, path, str(timestamp + '_phase' + str(phase) + '_history.txt'))

        previous_model_path = e2e_model_path

    utils.save_model(model, path)
    utils.class_predictions(model.e2e_model, steps_per_epoch, args.batch_size, args.img_size, 'test', test_df, val_imgs_preprocessed, val_idxs, val_labels, path, timestamp, phase)
    if(weakly):
        utils.compute_accuracy(model.e2e_model, model.num_classes, 'test', val_imgs_preprocessed, val_labels, path, timestamp, args.img_size, phase, masks=val_masks)
    else:
        utils.compute_accuracy(model.e2e_model, model.num_classes, 'test', val_imgs_preprocessed, val_labels, path, timestamp, args.img_size, phase)
    utils.save_explanations(model.e2e_model, steps_per_epoch, args.batch_size, args.img_size, 'test', test_df, val_imgs, val_imgs_preprocessed, val_idxs, path, timestamp, phase)
    
    del model
    print(gc.collect())

print('Finished')