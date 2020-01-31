from VGG import VGGClf
from ResNetMod import resnet50, resnet34, resnet18
from Explainer import Explainer
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
import utils
import Losses
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class ExplainerClassifierCNN(nn.Module):
    def __init__(self, exp_in_channels=3, exp_conv_filter_size=(3, 3), exp_pool_size=2,
                 dec_in_channels=3, dec_conv_filter_size=(3, 3), dec_pool_size=2,
                 img_size=(224, 224), dropout=0.3, num_classes=2, clf='VGG', init_bias=2.0, pretrained=False):
        super(ExplainerClassifierCNN, self).__init__()

        self.exp_in_channels = exp_in_channels
        self.exp_conv_filter_size = exp_conv_filter_size
        self.exp_pool_size = exp_pool_size

        self.dec_in_channels = dec_in_channels
        self.dec_conv_filter_size = dec_conv_filter_size
        self.dec_pool_size = dec_pool_size
        
        self.dropout = dropout
        self.img_size = img_size
        self.num_classes = num_classes  

        self.clf = clf
        self.init_bias = init_bias
        self.pretrained = pretrained

        if('resnet50' in self.clf.lower()):
            self.decmaker = resnet50(pretrained=self.pretrained, progress=True, img_size=img_size, num_classes=num_classes)
        elif('resnet34' in self.clf.lower()):
            self.decmaker = resnet34(pretrained=self.pretrained, progress=True, img_size=img_size, num_classes=num_classes)
        elif('resnet18' in self.clf.lower()):
            self.decmaker = resnet18(pretrained=self.pretrained, progress=True, img_size=img_size, num_classes=num_classes)
        else:
            self.decmaker = VGGClf(in_channels=self.dec_in_channels, 
                        dec_conv_filter_size=self.dec_conv_filter_size,
                        dec_pool_size=self.dec_pool_size, img_size=self.img_size,
                        dropout=self.dropout, num_classes=self.num_classes)

        self.explainer = Explainer(in_channels=self.exp_in_channels, 
                        exp_conv_filter_size=self.exp_conv_filter_size, 
                        exp_pool_size=self.exp_pool_size, 
                        img_size=self.img_size, init_bias=self.init_bias)

    def train(self, dataloader, optimizer, device, args, phase, weights, disable=False):
        self.decmaker.train()
        self.explainer.train()

        dec_criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weights) 

        if(phase == 0):
            utils.freeze(self.explainer)
            self.explainer.eval()
        elif(phase == 1):
            utils.unfreeze(self.explainer)
            utils.freeze(self.decmaker)
        elif(phase == 2):
            utils.unfreeze(self.decmaker)

        for batch_imgs, batch_labels, _, batch_masks in tqdm(dataloader, disable=disable):
            optimizer.zero_grad()
            if(len(batch_masks) > 0):
                batch_imgs, batch_labels, batch_masks = batch_imgs.to(device), batch_labels.to(device), batch_masks.to(device)
            else:
                batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

            batch_expls = self.explainer(batch_imgs)
            batch_probs = self.decmaker(batch_imgs, batch_expls)
            dec_loss = dec_criterion(batch_probs, batch_labels)

            if(args.loss == 'unsup'):
                exp_loss = Losses.batch_unsupervised_explanation_loss(batch_expls, float(args.beta), reduction='mean')
            elif(args.loss == 'weakly'):
                exp_loss = Losses.batch_weaklysupervised_explanation_loss(batch_expls, batch_masks, reduction='mean')
            elif(args.loss == 'hybrid'):
                exp_loss = Losses.batch_hybrid_explanation_loss(batch_expls, batch_masks, float(args.beta1), float(args.beta2), reduction='mean')
            
            loss = float(args.alfa) * dec_loss + (1-float(args.alfa)) * exp_loss

            loss.backward()
            optimizer.step()  
            

    def validation(self, dataloader, device, args, disable=False):
        self.decmaker.eval()
        self.explainer.eval()
        val_loss = 0
        val_exp_loss = 0
        val_dec_loss = 0
        val_acc = 0

        whole_preds = []
        whole_labels = []
        dec_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():

            for batch_imgs, batch_labels, _, batch_masks in tqdm(dataloader, disable=disable):
                
                if(len(batch_masks) > 0):
                    batch_imgs, batch_labels, batch_masks = batch_imgs.to(device), batch_labels.to(device), batch_masks.to(device)
                else:
                    batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
                batch_expls = self.explainer(batch_imgs)
                #print(batch_expls.mean(), batch_expls.std(), batch_expls.min(), batch_expls.max())

                batch_probs = self.decmaker(batch_imgs, batch_expls)
                batch_dec_loss = dec_criterion(batch_probs, batch_labels)

                if(args.loss == 'unsup'):
                    batch_exp_loss = Losses.batch_unsupervised_explanation_loss(batch_expls, float(args.beta), reduction='sum')
                elif(args.loss == 'weakly'):
                    batch_exp_loss = Losses.batch_weaklysupervised_explanation_loss(batch_expls, batch_masks, reduction='sum')
                elif(args.loss == 'hybrid'):
                    batch_exp_loss = Losses.batch_hybrid_explanation_loss(batch_expls, batch_masks, float(args.beta1), float(args.beta2), reduction='sum')

                batch_loss = float(args.alfa) * batch_dec_loss + (1-float(args.alfa)) * batch_exp_loss
                val_exp_loss += batch_exp_loss.item()
                val_dec_loss += batch_dec_loss.item()
                val_loss += batch_loss.item()
            

                batch_probs = F.softmax(batch_probs, dim=1)
                batch_max_probs, batch_preds = torch.max(batch_probs, 1)

                whole_labels.extend(batch_labels.data.cpu().numpy())
                whole_preds.extend(batch_preds.data.cpu().numpy())



            val_loss /= len(dataloader.dataset)            
            val_exp_loss /= len(dataloader.dataset)
            val_dec_loss /= len(dataloader.dataset)
            val_acc += accuracy_score(whole_labels, whole_preds)

        return val_loss, val_exp_loss, val_dec_loss, val_acc

    def test(self, dataloader, device, args, disable=False):
        self.decmaker.eval()
        self.explainer.eval()
        val_loss = 0
        val_exp_loss = 0
        val_dec_loss = 0
        val_acc = 0

        whole_preds = []
        whole_probs = []
        whole_labels = []
        dec_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():

            for batch_imgs, batch_labels, _, batch_masks in tqdm(dataloader, disable=disable):
                
                if(len(batch_masks) > 0):
                    batch_imgs, batch_labels, batch_masks = batch_imgs.to(device), batch_labels.to(device), batch_masks.to(device)
                else:
                    batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
                batch_expls = self.explainer(batch_imgs)

                batch_probs = self.decmaker(batch_imgs, batch_expls)
                batch_dec_loss = dec_criterion(batch_probs, batch_labels)

                if(args.loss == 'unsup'):
                    batch_exp_loss = Losses.batch_unsupervised_explanation_loss(batch_expls, float(args.beta), reduction='sum')
                elif(args.loss == 'weakly'):
                    batch_exp_loss = Losses.batch_weaklysupervised_explanation_loss(batch_expls, batch_masks, reduction='sum')
                elif(args.loss == 'hybrid'):
                    batch_exp_loss = Losses.batch_hybrid_explanation_loss(batch_expls, batch_masks, float(args.beta1), float(args.beta2), reduction='sum')

                batch_loss = float(args.alfa) * batch_dec_loss + (1-float(args.alfa)) * batch_exp_loss
                val_exp_loss += batch_exp_loss.item()
                val_dec_loss += batch_dec_loss.item()
                val_loss += batch_loss.item()
            

                batch_probs = F.softmax(batch_probs, dim=1)
                batch_max_probs, batch_preds = torch.max(batch_probs, 1)

                whole_labels.extend(batch_labels.data.cpu().numpy())
                whole_preds.extend(batch_preds.data.cpu().numpy())
                whole_probs.extend(batch_probs.data.cpu().numpy())


            val_loss /= len(dataloader.dataset)            
            val_exp_loss /= len(dataloader.dataset)
            val_dec_loss /= len(dataloader.dataset)
            val_acc += accuracy_score(whole_labels, whole_preds)

        return val_loss, val_exp_loss, val_dec_loss, val_acc, np.array(whole_probs), np.array(whole_preds), np.array(whole_labels)

    def save_explanations(self, dataloader, phase, device, path, thr=0.75, disable=False, test=False, epoch=-1):
        self.decmaker.eval()
        self.explainer.eval()
        print('\n SAVING EXPLANATIONS')
        timestamp = path.split('/')[-1]
        bid = 0
        with torch.no_grad():

            for batch_imgs, batch_labels, batch_names, _ in tqdm(dataloader, disable=disable):
                
                batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

                batch_expls = self.explainer(batch_imgs)

                batch_probs = self.decmaker(batch_imgs, batch_expls)
                
                for idx, img in enumerate(batch_imgs):
                    img = np.swapaxes(batch_imgs[idx].cpu().numpy(), 0, 2)
                    expl = np.swapaxes(batch_expls[idx].squeeze().cpu().numpy(), 0, 1)
                    
                    plt.figure(0)
                    plt.axis('off')
                    plt.imshow(expl, vmin=0., vmax=1.0, cmap='viridis')
                    if(epoch > 0):
                        #plt.savefig(os.path.join(path, '{}_phase{}_epoch{}_explanation_{}'.format(timestamp, str(phase), str(epoch), batch_names[idx])), bbox_inches='tight', transparent=True, pad_inches=0)
                        #plt.close()
                        return
                    else:
                        if(test):
                            plt.savefig(os.path.join(path, '{}_phase{}_explanation_test_{}'.format(timestamp, str(phase), batch_names[idx])), bbox_inches='tight', transparent=True, pad_inches=0)
                        else:
                            plt.savefig(os.path.join(path, '{}_phase{}_explanation_{}'.format(timestamp, str(phase), batch_names[idx])), bbox_inches='tight', transparent=True, pad_inches=0)
                        #np.savetxt(os.path.join(path, '{}_epoch{}_explanation_{}.txt'.format(timestamp, str(phase), batch_names[idx][:-4])), np.array(expl), newline='\n') 

                    plt.close()
                    #cv2.imwrite(os.path.join(path, '{}_phase{}_explanationCV2_{}'.format(timestamp, str(phase), batch_names[idx])), expl * 255)
                    
                    plt.figure(idx, figsize=(15, 5))
                    plt.subplot(131)
                    ax = plt.gca()
                    ax.relim()
                    ax.autoscale()
                    plt.title('Original Image')                                    
                    plt.imshow(img)

                    plt.subplot(132)
                    plt.title('Explanation')
                    plt.imshow(expl, vmin=0., vmax=1.0, cmap='viridis')

                    plt.subplot(133)
                    plt.imshow((expl > thr), vmin=0., vmax=1.0, cmap='viridis')
                    plt.title('Explanation > ' + str(thr))
                    if(test):
                        plt.savefig(os.path.join(path, '{}_phase{}_ex_img_thr{}_test_{}'.format(timestamp, str(phase), str(thr), batch_names[idx])))
                    else:
                        plt.savefig(os.path.join(path, '{}_phase{}_ex_img_thr{}_{}'.format(timestamp, str(phase), str(thr), batch_names[idx])))
                    plt.close()
                    
                bid += 1

    def checkpoint(self, filename, epoch, batch_loss, batch_acc, optimizer, save=True):

        # Update best loss
        if(not hasattr(self, 'best_loss')):
            self.best_loss = float('inf')
        if(not hasattr(self, 'best_acc')):
            self.best_acc = float(0)

        if self.best_loss > batch_loss:
            self.best_loss = batch_loss
        
        if self.best_acc < batch_acc:
            self.best_acc = batch_acc
        
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                    'best_loss': self.best_loss,
                    'best_acc': self.best_acc,
                    'decmaker': self.decmaker.module.state_dict() if type(
                        self.decmaker) is nn.parallel.DistributedDataParallel else self.decmaker.state_dict(),
                    'explainer': self.explainer.module.state_dict() if type(
                        self.explainer) is nn.parallel.DistributedDataParallel else self.explainer.state_dict(),
                    'optimizer': optimizer.state_dict()}

            # Save best checkpoint
            if self.best_loss == batch_loss:
                best_loss = filename + '_best_loss.pt'
                torch.save(chkpt, best_loss)
            
            if self.best_acc == batch_acc:
                best_acc = filename + '_best_acc.pt'
                torch.save(chkpt, best_acc)
            
            torch.save(chkpt, filename + '_latest.pt')

            # Delete checkpoint
            del chkpt