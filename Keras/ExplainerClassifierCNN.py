from sklearn.base import BaseEstimator, ClassifierMixin
from Explainer import Explainer
from VGG import VGGClf
from ResNet50Mod import ResNet50ModClf
from keras.utils import multi_gpu_model, plot_model
from keras import layers as KL
import numpy as np
np.random.seed(42)
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
import utils
import sys

class ExplainerClassifierCNN(BaseEstimator, ClassifierMixin):
    def __init__(self, exp_conv_num=3, exp_conv_conseq=2, exp_conv_filters=32,
                 exp_conv_filter_size=(3, 3), exp_conv_activation='relu',
                 exp_pool_size=2,
                 dec_conv_num=4, dec_conv_conseq=2, dec_conv_filters=32,
                 dec_conv_filter_size=(3, 3), dec_conv_activation='relu', 
                 dec_pool_size=2,
                 dec_dense_num=2, dec_dense_width=128, dec_dense_activation='sigmoid',
                 img_size=(224, 224),
                 dropout=0.3, batch_size=32,
                 num_classes=2, nr_gpus=1,
                 clf='VGG', loss='weakly',
                ):

        self.exp_conv_num = exp_conv_num
        self.exp_conv_conseq = exp_conv_conseq
        self.exp_conv_filters = exp_conv_filters
        self.exp_conv_filter_size = exp_conv_filter_size
        self.exp_conv_activation = exp_conv_activation
        self.exp_pool_size = exp_pool_size

        self.dec_conv_num = dec_conv_num
        self.dec_conv_conseq = dec_conv_conseq
        self.dec_conv_filters = dec_conv_filters
        self.dec_conv_filter_size = dec_conv_filter_size
        self.dec_conv_activation = dec_conv_activation
        self.dec_pool_size = dec_pool_size
        
        self.dec_dense_num = dec_dense_num
        self.dec_dense_width = dec_dense_width
        self.dec_dense_activation = dec_dense_activation
        
        self.dropout = dropout
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes  
        self.nr_gpus = nr_gpus

        self.clf = clf
        self.loss = loss
        if(self.clf == 'ResNet50Mod'):
            self.decmaker = ResNet50ModClf(self.img_size, self.num_classes)
        else:
            self.decmaker = VGGClf(dec_conv_num=self.dec_conv_num, dec_conv_conseq=self.dec_conv_conseq, 
                 dec_conv_filters=self.dec_conv_filters,
                 dec_conv_filter_size=self.dec_conv_filter_size, dec_conv_activation=self.dec_conv_activation, 
                 dec_pool_size=self.dec_pool_size, dec_dense_num=self.dec_dense_num, dec_dense_width=self.dec_dense_width, 
                 dec_dense_activation=self.dec_dense_activation, img_size=self.img_size,
                 dropout=self.dropout, num_classes=self.num_classes)


    def build_model(self, phase, pretrained):
        
        self.explainer = Explainer()
        self.explainer.build_explainer()
          
        if(isinstance(self.decmaker, ResNet50ModClf)):
            self.decmaker.build_classifier(phase, pretrained)
        else:
            self.decmaker.build_classifier()

        input_image = KL.Input(tuple(list(self.img_size) + [3]), name='input_img')
        fixed_explanation_input =  KL.Input(tensor=K.constant(np.ones(tuple([self.batch_size] + list(self.img_size) + [1]))), name='fixed_explanation_input')

        explanation = self.explainer.model(input_image)
        
        if(phase < 2):
            decision = self.decmaker.model([fixed_explanation_input, input_image])
            self.e2e_model = Model(inputs=[input_image, fixed_explanation_input], outputs=[explanation, decision])
        else: 
            decision = self.decmaker.model([explanation, input_image])
            self.e2e_model = Model(inputs=[input_image], outputs=[explanation, decision])
        
        if(self.nr_gpus > 1):
            self.e2e_model_gpu = multi_gpu_model(self.e2e_model, self.nr_gpus)
        else:
            self.e2e_model_gpu = self.e2e_model

        return self
        
    def generate_generator(self, dims, gen):
        while True:
            imgs, labels = next(gen)
            yield imgs, {'decision-maker': labels, 'explainer': np.zeros(dims)}

    def fit(self, tr_df, val_df, nr_epochs, steps_per_epoch, callbacks, path, augment=False, loss='unsup'):
        if(nr_epochs > 0):
            if(augment):
                save_path = os.path.join(path, 'augmented')
                if(not os.path.exists(save_path)):
                    os.makedirs(save_path)
                print(save_path)
                datagen_args = dict(
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            preprocessing_function=self.decmaker.preprocess_function,
                            rescale=1./255)
                            
                image_datagen = ImageDataGenerator(**datagen_args)

                train_generator = image_datagen.flow_from_dataframe(
                        dataframe=tr_df,
                        x_col='imageID',
                        y_col='GT_real',
                        target_size=self.img_size,
                        batch_size=self.batch_size,
                        class_mode='categorical',
                        shuffle=True,
                        seed=42,
                        save_prefix='aug',
                        save_to_dir=save_path)
                
                
                
                if((loss == 'weakly') or (loss == 'hybrid')):
                    mask_datagen = ImageDataGenerator(**datagen_args)
                    mask_generator = mask_datagen.flow_from_dataframe(
                        dataframe=tr_df,
                        x_col='maskID',
                        class_mode=None,
                        shuffle=True,
                        seed=42,
                        target_size=self.img_size,
                        batch_size=self.batch_size,
                        save_prefix='aug_mask',
                        save_to_dir=save_path)
                    tr_gen = zip(image_generator, mask_generator)
                else:                    
                    t = tuple([self.batch_size] + list(self.img_size) + [3])
                    tr_gen = self.generate_generator(t, train_generator)
                    #tr_gen = zip(train_generator, z)
                    #print(next(tr_gen))
                    #sys.exit(0)
            else:
                tr_gen = utils.image_generator(tr_df, batch_size=self.batch_size, img_size=self.img_size,
                                        num_classes=self.num_classes, preprocess_function=self.decmaker.preprocess_function)
                #print(next(tr_gen))
                #sys.exit(0)
            
            val_gen = utils.image_generator(val_df, batch_size=self.batch_size, img_size=self.img_size,
                                        num_classes=self.num_classes, preprocess_function=self.decmaker.preprocess_function)

            self.e2e_model_gpu.fit_generator(tr_gen, validation_data=val_gen,
                                    epochs=nr_epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_steps=steps_per_epoch,
                                    callbacks=callbacks,
                                    verbose=True, use_multiprocessing=False)
        return self
    
    def save_architecture(self, phase, timestamp, path, option=None, layer=None):
        if((option is not None) and (layer is not None)):
            self.exp_model_filename = timestamp + '_phase' + str(phase) + '_' + option + '_layer_' + str(layer) + '_model_exp.png'
            self.dec_model_filename = timestamp + '_phase' + str(phase) + '_' + option + '_layer_' + str(layer) + '_model_dec.png'
            self.e2e_model_filename = timestamp + '_phase' + str(phase) + '_' + option + '_layer_' + str(layer) + '_model_e2e.png'
        else:
            self.exp_model_filename = timestamp + '_phase' + str(phase) + '_model_exp.png'
            self.dec_model_filename = timestamp + '_phase' + str(phase) + '_model_dec.png'
            self.e2e_model_filename = timestamp + '_phase' + str(phase) + '_model_e2e.png'
            
        plot_model(self.explainer.model, to_file=os.path.join(path, self.exp_model_filename))
        print("Model printed to " + os.path.join(path, self.exp_model_filename))

        plot_model(self.decmaker.model, to_file=os.path.join(path, self.dec_model_filename))
        print("Model printed to " + os.path.join(path, self.dec_model_filename))

        plot_model(self.e2e_model, to_file=os.path.join(path, self.e2e_model_filename))
        print("Model printed to " + os.path.join(path, self.e2e_model_filename))
