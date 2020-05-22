#!/usr/bin/env python
# coding: utf-8
import os
import sys
from ExplainerClassifierCNN import ExplainerClassifierCNN
import utils
import argparse
import json
import numpy as np
np.random.seed(42)
import random as rn
rn.seed(1234)
from tensorflow.python.client import device_lib

from keras import backend as K
from keras.models import load_model
from keras.applications import imagenet_utils


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--gpu", type=str, default='1',
                    help="Which gpus to use in CUDA_VISIBLE_DEVICES.")                  
parser.add_argument("dataset", type=str,
                    help="Dataset to load.")
parser.add_argument("--dataset_path", type=str, default="/media/icrto/TOSHIBA6T/ICRTO",
                    help="Folder where dataset is located.")                    
parser.add_argument("model_folder", type=str, 
                    help="Folder where the model is stored.")
parser.add_argument("--beta", type=float,
                    help="Lexplic_unsup = beta * L1 + (1-beta) * Total Variation")    
parser.add_argument("--beta1", type=float,
                    help="Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss")  
parser.add_argument("--beta2", type=float,
                    help="Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss")  
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size.")   
parser.add_argument("--phase", type=int, default=2,
                    help="Test model from which phase.")   
parser.add_argument("--sanity_checks", default=False,
                    help="Apply sanity checks.")
parser.add_argument("--option", type=str,
                    help="Option for sanity checks.")  
parser.add_argument("--layer", type=int, 
                    help="Layer being analysed for sanity checks.")   
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# confirm TensorFlow sees the GPU
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
assert len(K.tensorflow_backend._get_available_gpus()) > 0


json_file = os.path.join(args.model_folder, 'model.json')    
if(args.sanity_checks):
    if(args.option == None or args.layer == None):
        print('Please specify the option and layer when performing a sanity check.')
        sys.exit(-1)
    timestamp = args.model_folder.split('/')[-3][:19]
    model_file = os.path.join(args.model_folder, str(timestamp + '_phase' + str(args.phase) + '_' + args.option + '_layer_' + str(args.layer) + '_model.h5'))
else:
    timestamp = args.model_folder.split('/')[-1][:19]
    model_file = os.path.join(args.model_folder, str(timestamp + '_phase' + str(args.phase) + '_model.h5'))


with open(json_file) as json_file:
    model_args = json.loads(json.load(json_file))
print("Loaded model args from disk")

loss = model_args['loss']
weakly = False
if(loss == 'weakly' or loss == 'hybrid'):
    weakly = True
elif(loss == 'unsup' and args.beta == None):
    print('Please specify the beta value when using an unsupervised loss function.')
    sys.exit(-1)
elif(loss == 'hybrid' and (args.beta1 == None or args.beta2 == None)):
    print('Please specify the beta1 and beta2 values when using an hybrid loss function.')
    sys.exit(-1)

dtset = args.dataset
_, _, test_df = utils.load_data(dtset, path=args.dataset_path, weakly=weakly)
steps_per_epoch = int(np.ceil(len(test_df)/args.batch_size))

unsup_loss = utils.unsupervised_explanation_loss(beta=args.beta)
hybrid_loss = utils.hybrid_explanation_loss(beta1=args.beta1, beta2=args.beta2)
e2e_model = load_model(model_file, custom_objects={"active_pixels":utils.active_pixels, 
    "weaklysupervised_explanation_loss":utils.weaklysupervised_explanation_loss, 
    unsup_loss.__name__:unsup_loss, 
    hybrid_loss.__name__:hybrid_loss
})
if(model_args['clf'] == 'VGG'):
    preprocess_function = utils.preprocess
elif(model_args['clf'] == 'ResNet50Mod'):
    preprocess_function = imagenet_utils.preprocess_input
if(weakly):
    imgs, imgs_preprocessed, idxs, labels, masks = utils.get_images_and_labels(tuple(model_args['img_size']), test_df, preprocess_function, weakly)
else:
    imgs, imgs_preprocessed, idxs, labels = utils.get_images_and_labels(tuple(model_args['img_size']), test_df, preprocess_function, weakly)


utils.class_predictions(e2e_model, steps_per_epoch, args.batch_size, model_args['img_size'], 'test', test_df, imgs_preprocessed, idxs, labels, args.model_folder, timestamp, args.phase, args.option, args.layer)
if(weakly):
    utils.compute_accuracy(e2e_model, model_args['num_classes'], 'test', imgs_preprocessed, labels, args.model_folder, timestamp, model_args['img_size'], args.phase, args.option, args.layer, masks)
else:
    utils.compute_accuracy(e2e_model, model_args['num_classes'], 'test', imgs_preprocessed, labels, args.model_folder, timestamp, model_args['img_size'], args.phase, args.option, args.layer)
utils.save_explanations(e2e_model, steps_per_epoch, args.batch_size, model_args['img_size'], 'test', test_df, imgs, imgs_preprocessed, idxs, args.model_folder, timestamp, args.phase, args.option, args.layer)

print('Finished')


