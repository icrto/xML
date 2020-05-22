import os
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset as dtset
import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import utils
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
parser.add_argument('model_ckpt', type=str, 
                    help='Model to load.')
parser.add_argument('--loss', type=str, default='weakly',
                    help='Specifiy if loss is weakly/hybrid/unsupervised or not.')
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
parser.add_argument('--class_weights', action='store_true', default=False, help='Use class weighting in loss function.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = os.path.dirname(args.model_ckpt)

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


model = ExplainerClassifierCNN(num_classes=args.nr_classes,
                dec_conv_filter_size=args.dec_conv_filter_size, dec_pool_size=args.dec_pool_size, img_size=args.img_size,
                dropout=args.dropout, clf=args.classifier, dec_in_channels=args.dec_in_channels, exp_in_channels=args.exp_in_channels,
                exp_conv_filter_size=args.exp_conv_filter_size, exp_pool_size=args.exp_pool_size, init_bias=args.init_bias)


ckpt = torch.load(args.model_ckpt, map_location=device)
ckpt_epoch = ckpt['epoch']
ckpt_loss = ckpt['best_loss']
ckpt_acc = ckpt['best_acc']

model.decmaker.load_state_dict(ckpt['decmaker'])
model.explainer.load_state_dict(ckpt['explainer'])
model.to(device)

if(args.class_weights):
    class_weights = 'balanced'
else: 
    class_weights = None

_, _, test_df, _ = Dataset.load_data(folder=args.dataset_path, dataset=args.dataset, test=True, weakly=weakly, cropped=args.cropped, class_weights=class_weights)
test_dataset = dtset(test_df, weakly=weakly, img_size=args.img_size, augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

test_global_loss, test_explainer_loss, test_decmaker_loss, test_decmaker_acc, whole_probs, whole_preds, whole_labels = model.test(test_loader, device, args)

if('imagenet' in args.dataset):
    classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
else:
    classes = ['neg', 'pos']

timestamp = path.split('/')[-1]
if(args.nr_classes > 2):
    utils.plot_roc_curve_multiclass(os.path.join(path, timestamp), whole_probs, whole_labels, classes)
    utils.plot_precision_recall_curve_multiclass(os.path.join(path, timestamp), whole_probs, whole_labels, classes)
else:
    utils.plot_roc_curve(os.path.join(path, timestamp), whole_probs, whole_labels, classes)
    utils.plot_precision_recall_curve(os.path.join(path, timestamp), whole_probs, whole_labels, classes)
print('Test Loss %f\tTest Exp Loss %f\tTest Dec Loss %f\tTest Acc %f' % (test_global_loss, test_explainer_loss, test_decmaker_loss, test_decmaker_acc))
print()
with open(os.path.join(path, 'test_stats_best_loss.txt'), 'w') as f:
    print('Epoch %f\tCkpt Loss %f\tCkpt Acc %f\tTest Loss %f\tTest Exp Loss %f\tTest Dec Loss %f\tTest Acc %f' % (ckpt_epoch, ckpt_loss, ckpt_acc, test_global_loss, test_explainer_loss, test_decmaker_loss, test_decmaker_acc), file=f)
    #print('Epoch %f\tCkpt Loss %f\tTest Loss %f\tTest Exp Loss %f\tTest Dec Loss %f\tTest Acc %f' % (ckpt_epoch, ckpt_loss, test_global_loss, test_explainer_loss, test_decmaker_loss, test_decmaker_acc), file=f)

#model.save_explanations(test_loader, 2, device, path, test=True)
