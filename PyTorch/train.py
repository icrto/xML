import os
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset as dtset
import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import argparse
import utils
import csv
from EarlyStopping import EarlyStopping
from summary import summary

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Args.')
parser.add_argument('--gpu', type=str, default='1',
                    help='Which gpus to use in CUDA_VISIBLE_DEVICES.')                 
parser.add_argument('dataset', type=str,
                    help='Dataset to load.')
parser.add_argument('--dataset_path', type=str, default='/media/TOSHIBA6T/ICRTO',
                    help='Folder where dataset is located.')                  
parser.add_argument('--nr_classes', type=int, default=2,
                    help='Number of target classes.')
parser.add_argument('nr_epochs', type=str,
                    help='Number of epochs for each of the 3 training phases as an array, for example: 50,100,50.')
parser.add_argument('-bs', '--batch_size', type=int, default=32,
                    help='Training batch size.')
parser.add_argument('alfa', type=str,
                    help='Loss = alfa * Lclassif + (1-alfa) * Lexplic')                   
parser.add_argument('--beta', type=str,
                    help='Lexplic_unsup = beta * L1 + (1-beta) * Total Variation')    
parser.add_argument('--beta1', type=str,
                    help='Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss')  
parser.add_argument('--beta2', type=str,
                    help='Lexplic_hybrid = beta1 * L1 + beta2 * Total Variation + (1-beta1-beta2)* Weakly Loss')  
parser.add_argument('folder', type=str, 
                    help='Directory where images and models are to be stored.')
parser.add_argument('--pretrained', default=False, action='store_true',
                    help='True if one wants to load pre trained models in training phase 1, False otherwise.')
parser.add_argument('--loss', type=str, default='weakly',
                    help='Specifiy if loss is weakly/hybrid/unsupervised or not.')
parser.add_argument('--opt', type=str, default='adadelta', 
                    help='Optimizer to use. Either adadelta or sgd.')
parser.add_argument('-lr', '--learning_rate', type=str,
                    help='Learning rate for each training phase.')
parser.add_argument('--decay', type=float, default=0.0001,
                    help='Learning rate decay to use with sgd.')
parser.add_argument('-mom', '--momentum', type=float, default=0.9,
                    help='Momentum to use with sgd.')
parser.add_argument('-min_lr', '--min_learning_rate', type=float, default=1e-5,
                    help='Minimum learning rate to use with sgd and with ReduceLearningRateonPlateau.')
parser.add_argument('--patience', type=int, default=10,
                    help='Patience (number of epochs for a model to be considered as converged) to use with sgd and with ReduceLearningRateonPlateau.')
parser.add_argument('--factor', type=float, default=0.2, 
                    help='Learning rate changing factor to use with sgd and with ReduceLearningRateonPlateau.')
parser.add_argument('--exp_in_channels', type=int, default=3,
                    help="Number of input channels of the explainer.")      
parser.add_argument('--exp_conv_filter_size', type=tuple, default=(3,3),
                    help="Filter size for the initial explainer stage's filters.")
parser.add_argument('--exp_pool_size', type=int, default=2,
                    help="Explainer pooling layers' size.")
parser.add_argument('--dec_in_channels', type=int, default=3,
                    help="Number of input channels of the decision maker.")                    
parser.add_argument('--dec_conv_filter_size', type=tuple, default=(3,3),
                    help="Filter size for the initial explainer transpose convolutional stage's filters.")
parser.add_argument('--dec_pool_size', type=int, default=2,
                    help="Explainer pooling layers' size - transpose convolution.")
parser.add_argument('--img_size', type=tuple, default=(224, 224), help='Input image size.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (VGG classifier).')
parser.add_argument('--init_bias', type=float, default=1.0, help='Initial bias for convolutional layers of the Explainer.')
parser.add_argument('-clf', '--classifier', type=str, default='VGG', help='Classifier (VGG or ResNet50).')
parser.add_argument('--cropped', action='store_true', default=False, help='Load NIH-NCI dataset with cropped images if True, otherwise False.')
parser.add_argument('--augmentation', action='store_true', default=False, help='Perform data augmentation.')
parser.add_argument('--aug_prob',type=float, default=0.8, help='Probability of applying data augmentation to each image. Only relevant if augmentation == True.')
parser.add_argument('--early_patience', type=str, default='200,200,200', help='Number of epochs to consider before Early Stopping.')
parser.add_argument('--early_delta', type=int, default=1e-4, help='Minimum change in the monitored quantity to qualify as an improvement for Early Stopping.')
parser.add_argument('--class_weights', action='store_true', default=False, help='Use class weighting in loss function.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

loss = args.loss
weakly = False
if(loss == 'weakly' or loss == 'hybrid'):
    weakly = True

if(loss == 'unsup'):
    if(args.beta is None):
        print('Please define a value for beta.')
        sys.exit(-1)
elif(loss == 'weakly'):
    pass
elif(loss == 'hybrid'):
    if(args.beta1 is None):
        print('Please define a value for beta1.')
        sys.exit(-1)
    if(args.beta2 is None):
        print('Please define a value for beta2.')
        sys.exit(-1)
    if(args.beta1 + args.beta2 >= 1.0):
        print('Beta1 and beta2 must be such that beta1 + beta2 < 1.0.')
        sys.exit(-1)
else:
    print('Invalid loss function. Try again with <unsup>, <weakly> or <hybrid>.')
    sys.exit(-1)

eps = args.nr_epochs.split(',')
nr_epochs = np.array([int(x) for x in eps])

lrs = args.learning_rate.split(',')
lr = np.array([float(x) for x in lrs])

folder = args.folder
timestamp, path = utils.create_folder(folder) 

with open(os.path.join(path, timestamp + '_train_parameters_summary.txt'), 'w') as f:
    f.write(str(args))

model = ExplainerClassifierCNN(num_classes=args.nr_classes,
                dec_conv_filter_size=args.dec_conv_filter_size, dec_pool_size=args.dec_pool_size, img_size=args.img_size,
                dropout=args.dropout, clf=args.classifier, dec_in_channels=args.dec_in_channels, exp_in_channels=args.exp_in_channels,
                exp_conv_filter_size=args.exp_conv_filter_size, exp_pool_size=args.exp_pool_size, init_bias=args.init_bias, pretrained=args.pretrained)
model.to(device)

summary(model.decmaker, [(3, 224, 224), (1, 224, 224)], filename=os.path.join(path, timestamp + '_model_dec_info.txt'))
summary(model.explainer, (3, 224, 224), filename=os.path.join(path, timestamp + '_model_exp_info.txt'))

if(args.class_weights):
    class_weights = 'balanced'
else: 
    class_weights = None

tr_df, val_df, test_df, weights = Dataset.load_data(folder=args.dataset_path, dataset=args.dataset, test=False, weakly=weakly, cropped=args.cropped, class_weights=class_weights)
weights = torch.FloatTensor(weights).to(device)

train_dataset = dtset(tr_df, weakly=weakly, img_size=args.img_size, augmentation=args.augmentation, aug_prob=args.aug_prob)
val_dataset = dtset(val_df, weakly=weakly, img_size=args.img_size, augmentation=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

early_patience = args.early_patience.split(',')
early_patience = np.array([int(x) for x in early_patience])
for phase in range(3):
    print('PHASE ', str(phase))
    if(nr_epochs[phase] == 0): continue

    early_stopping = EarlyStopping(patience=early_patience[phase], delta=args.early_delta, verbose=True, folder=path, timestamp=timestamp)

    history_file = os.path.join(path, timestamp + '_phase' + str(phase) + '_history.csv')
    checkpoint_filename = os.path.join(path, timestamp + '_phase' + str(phase))

    with open(history_file, 'a+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(['epoch', 'train_global_loss', 'train_decmaker_loss', 'train_explainer_loss', 'train_decmaker_acc', 'val_global_loss', 'val_decmaker_loss', 'val_explainer_loss', 'val_decmaker_acc'])    

    if(args.opt == 'adadelta'):
        opt = optim.Adadelta(model.parameters(), lr=lr[phase], weight_decay=args.decay)
    else:
        opt = optim.SGD(model.parameters(), lr=lr[phase], weight_decay=args.decay, momentum=args.momentum)

    if('resnet' in args.classifier.lower()): # and 'imagenet' in args.dataset.lower()):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=args.factor, patience=args.patience, verbose=True, min_lr=args.min_learning_rate)
    else:
        scheduler = None

    for epoch in range(nr_epochs[phase]): 
        print('Epoch %d / %d' % (epoch, nr_epochs[phase]))
        print('Train')
        model.train(train_loader, opt, device, args, phase, weights)
        print('Val')
        train_global_loss, train_explainer_loss, train_decmaker_loss, train_decmaker_acc = model.validation(train_loader, device, args)
        print('Train Loss %f \tTrain Exp Loss %f \tTrain Dec Loss %f \tTrain Acc %f' % (train_global_loss, train_explainer_loss, train_decmaker_loss, train_decmaker_acc))
        val_global_loss, val_explainer_loss, val_decmaker_loss, val_decmaker_acc = model.validation(val_loader, device, args)
        print('Val Loss %f \tVal Exp Loss %f \tVal Dec Loss %f \tVal Acc %f' % (val_global_loss, val_explainer_loss, val_decmaker_loss, val_decmaker_acc))
        print()
        model.checkpoint(checkpoint_filename, epoch, val_global_loss, val_decmaker_acc, opt)

        with open(history_file, 'a+') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow([epoch, train_global_loss, train_decmaker_loss, train_explainer_loss, train_decmaker_acc, val_global_loss, val_decmaker_loss, val_explainer_loss, val_decmaker_acc])    

        early_stopping(val_global_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if(scheduler):
            if(isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(val_global_loss)
        
    print()
    utils.plot_metric_train_val(epoch + 1, history_file, 'decmaker_loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_decmaker_loss.png'), 'Classifier Loss')
    utils.plot_metric_train_val(epoch + 1, history_file, 'explainer_loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_explainer_loss.png'), 'Explainer Loss')
    utils.plot_metric_train_val(epoch + 1, history_file, 'global_loss', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_global_loss.png'), 'Global Loss')
    utils.plot_metric_train_val(epoch + 1, history_file, 'decmaker_acc', path, os.path.join(path, timestamp + '_phase' + str(phase) + '_decmaker_acc.png'), 'Accuracy')
    
    model.save_explanations(val_loader, phase, device, path, epoch=-1)
