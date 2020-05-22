from sklearn.base import BaseEstimator, ClassifierMixin
import utils
from keras import layers as KL
from keras.models import Model

class VGGClf(BaseEstimator, ClassifierMixin):
    def __init__(self, dec_conv_num=4, dec_conv_conseq=2, dec_conv_filters=32,
                 dec_conv_filter_size=(3, 3), dec_conv_activation='relu', 
                 dec_pool_size=2,
                 dec_dense_num=2, dec_dense_width=128, dec_dense_activation='sigmoid',
                 img_size=(224, 224),
                 dropout=0.3, num_classes=2
                 ):

    
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
        self.img_size = img_size
        self.num_classes = num_classes
        
        self.preprocess_function = utils.preprocess


    def build_classifier(self):
        input_explanation_layer = KL.Input(tuple(list(self.img_size) + [1]),
                                           name='decision-explanation-input')
       

        input_image_layer = KL.Input(tuple(list(self.img_size) + [3]),
                                     name='decision-image-input')
        
        last_expl = input_explanation_layer
        last_dec = input_image_layer
        
        #last_dec = KL.Multiply()([last_expl, last_dec])
        #last_dec = KL.Multiply()([KL.Concatenate()([last_expl] * 3), last_dec])
        
        
        # Convolutional section
        for conv_level in range(self.dec_conv_num):
            nfilters = self.dec_conv_filters * 2 ** conv_level
            
            # Convolutional layers
            for c in range(self.dec_conv_conseq):
                cfs = self.dec_conv_filter_size

                conv_dec = KL.Conv2D(nfilters, cfs,
                                     padding='same',
                                     activation=self.dec_conv_activation,
                                     name='decmaker-image-conv%d-%d' % (conv_level, c))
                last_dec = conv_dec(last_dec)

            last_dec = KL.Multiply()([#last_expl,
                                     KL.Concatenate()([last_expl] * nfilters),
                                      last_dec])

            # Pooling layer
            # neg = KL.Lambda(lambda x: -x)
            last_expl = KL.MaxPooling2D(pool_size=(self.dec_pool_size, self.dec_pool_size),
                                        name='decmaker-explanation-pool%d' % conv_level)(last_expl)
            last_dec = KL.MaxPooling2D(pool_size=(self.dec_pool_size, self.dec_pool_size),
                                       name='decmaker-image-pool%d' % conv_level)(last_dec)

        
        #last_dec = KL.Multiply()([#last_expl,
                                  #KL.Concatenate()([last_expl] * nfilters),
                                  #last_dec])
        
        last_dec = KL.GlobalMaxPool2D()(last_dec)
        
        for d in range(self.dec_dense_num):
            last_dec = KL.Dense(self.dec_dense_width, activation=self.dec_dense_activation)(last_dec)
            
            if self.dropout is not None and self.dropout > 0:
                last_dec = KL.Dropout(rate=self.dropout, seed=42)(last_dec)
        
        last_dec = KL.Dense(self.num_classes, activation='softmax')(last_dec)
        
        self.model = Model(inputs=[input_explanation_layer, input_image_layer],
                         outputs=[last_dec], name='decision-maker')
        
        
        #decmaker.summary()
        return self
    
    def freeze(self):
        for layer in self.model.layers:
            layer.trainable = False 
        return self
          
