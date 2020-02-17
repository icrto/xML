import os
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset as dtset
import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
from tqdm import tqdm
import csv
import cv2
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
parser.add_argument('--dec_in_channels', type=int, default=3,
                    help="Number of input channels of the decision maker.")                    
parser.add_argument('--dec_conv_filter_size', type=tuple, default=(3,3),
                    help="Filter size for the initial explainer transpose convolutional stage's filters.")
parser.add_argument('--dec_pool_size', type=int, default=2,
                    help="Explainer pooling layers' size - transpose convolution.")
parser.add_argument('--exp_in_channels', type=int, default=3,
                    help="Number of input channels of the explainer.")      
parser.add_argument('--exp_conv_filter_size', type=tuple, default=(3,3),
                    help="Filter size for the initial explainer stage's filters.")
parser.add_argument('--exp_pool_size', type=int, default=2,
                    help="Explainer pooling layers' size.")
parser.add_argument('--img_size', type=tuple, default=(224, 224), help='Input image size.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (VGG classifier).')
parser.add_argument('-clf', '--classifier', type=str, default='VGG', help='Classifier (VGG or ResNet50).')
parser.add_argument('--cropped', action='store_true', default=False, help='Load NIH-NCI dataset with cropped images if True, otherwise False.')
parser.add_argument('--class_weights', action='store_true', default=False, help='Use class weighting in loss function.')
parser.add_argument('--patch_size', type=int, default=16, help="Size of the patch to perform the occlusion.")
parser.add_argument('--thr', type=float, default=0.75, help="Threshold for explanations.")
parser.add_argument('--init_bias', type=float, default=1.0, help='Initial bias for convolutional layers of the Explainer.')


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

model.decmaker.load_state_dict(ckpt['decmaker'])
model.explainer.load_state_dict(ckpt['explainer'])
model.to(device)
model.decmaker.eval()
model.explainer.eval()


if(args.class_weights):
    class_weights = 'balanced'
else: 
    class_weights = None

_, _, test_df, _ = Dataset.load_data(folder=args.dataset_path, dataset=args.dataset, test=True, cropped=args.cropped, class_weights=class_weights)
test_dataset = dtset(test_df, img_size=args.img_size)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)


timestamp = path.split('/')[-1]
stats_file = os.path.join(path, 'clf_stats.csv')
columns = ['img', 'label', 'original']
steps = int(np.floor(args.img_size[0] / args.patch_size)) ** 2
steps = list(range(steps))
columns.extend(steps)

with open(stats_file, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(columns)    

for img, label, name, _ in tqdm(test_loader, disable=False):
    img, label = img.to(device), label.to(device)
    img_probs = []
    img_probs.append(name[0])
    img_probs.append(label.squeeze().cpu().detach().numpy())

    with torch.no_grad():
        
        expl = model.explainer(img)
        og_prob = model.decmaker(img, expl)
        og_prob = F.softmax(og_prob, dim=1)
        img_probs.append(og_prob.squeeze().cpu().detach().numpy()[1])
        
        img_np = np.swapaxes(batch_imgs[idx].cpu().numpy(), 0, 2)
        expl = np.swapaxes(batch_expls[idx].squeeze().cpu().numpy(), 0, 1)

        # H W C
        width = expl.shape[1]
        height = expl.shape[0]
        patches_relevance = []
        for row in range(int(np.floor(height / args.patch_size))):
            for col in range(int(np.floor(width/ args.patch_size))):
                startx = col * args.patch_size
                starty = row * args.patch_size
                widthx = int(min(args.patch_size, args.patch_size - ((col + 1) * args.patch_size - width)))
                heighty = int(min(args.patch_size, args.patch_size - ((row + 1) * args.patch_size - height)))
                patch = expl[starty:starty+heighty, startx:startx+widthx]
                patches_relevance.append((patch.mean(), starty, startx))

        patches_relevance.sort(reverse=True)

        new_img = img_np.copy()

        for patch in patches_relevance:
            _, starty, startx = patch[0], patch[1], patch[2]
            new_img[starty:starty+heighty, startx:startx+widthx, :] = 1
            
            new_img_tensor = np.transpose(new_img, (2,0,1))
            new_img_tensor = new_img_tensor[None, :, :, :]
            new_img_tensor = torch.from_numpy(new_img_tensor).float().to(device)
            
            expl_occluded = model.explainer(new_img_tensor)
            prob = model.decmaker(img, expl_occluded)
            prob = F.softmax(prob, dim=1)
            img_probs.append(prob.squeeze().cpu().detach().numpy()[1])
        
        with open(stats_file, 'a+') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(img_probs)    



