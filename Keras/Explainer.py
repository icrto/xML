from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Model
from keras import layers as KL

class Explainer(BaseEstimator):
    def __init__(self, exp_conv_num=3, exp_conv_conseq=2, exp_conv_filters=32,
                 exp_conv_filter_size=(3, 3), exp_conv_activation='relu',
                 exp_pool_size=2,
                 img_size=(224, 224)
                 ):

        self.exp_conv_num = exp_conv_num
        self.exp_conv_conseq = exp_conv_conseq
        self.exp_conv_filters = exp_conv_filters
        self.exp_conv_filter_size = exp_conv_filter_size
        self.exp_conv_activation = exp_conv_activation
        self.exp_pool_size = exp_pool_size

        self.img_size = img_size


    def build_explainer(self):
            input_layer = KL.Input(tuple(list(self.img_size) + [3]), name='explainer-input')
            last = input_layer

            # Convolutional section
            last_conv_per_level = []
            sizes = []
            for conv_level in range(self.exp_conv_num):
                nfilters = self.exp_conv_filters * 2 ** conv_level
                sizes.append(nfilters)
                
                # Convolutional layers
                for c in range(self.exp_conv_conseq):
                    cfs = self.exp_conv_filter_size

                    last = KL.Conv2D(nfilters, cfs,
                                    activation=self.exp_conv_activation,
                                    padding='same',
                                    name='explainer-conv%d-%d' % (conv_level, c))(last)

                last_conv_per_level.append(last)

                # Pooling layer
                if conv_level != self.exp_conv_num:
                    last = KL.MaxPooling2D(pool_size=(self.exp_pool_size, self.exp_pool_size),
                                        name='explainer-pool%d' % conv_level)(last)

            # Deconvolutional section
            for conv_level in range(self.exp_conv_num)[::-1]:
                cc = KL.Add(name='explainer-add%d' % conv_level)

                last = cc([KL.Conv2DTranspose(sizes[conv_level],
                                            self.exp_pool_size,
                                            strides=(self.exp_pool_size,
                                                    self.exp_pool_size),
                                            name='explainer-transpose%d-%d' % (conv_level, 0)
                                            )(last),
                        last_conv_per_level[conv_level]])

                for c in range(self.exp_conv_conseq):
                    cfs = self.exp_conv_filter_size

                    last = KL.Conv2D(sizes[conv_level], cfs,
                                    activation=self.exp_conv_activation,
                                    padding='same',
                                    name='explainer-deconv%d-%d' % (conv_level, c)
                                    )(last)

            last = KL.Conv2D(1, (1, 1), activation='linear', 
                            #activity_regularizer=regularizers.l1(1e-6),
                            name='explainer-conv-output')(last)
            last = KL.BatchNormalization()(last)
            last = KL.Activation('tanh')(last)
            last = KL.Activation('relu')(last)
            #last = KL.Lambda(lambda x: K.maximum(0., x - 0.0))(last)
            #KL.Activation('relu')(last)
            #last = KL.Activation('tanh')(last)
            out = last
            
            self.model = Model(inputs=[input_layer], outputs=[out], name='explainer')

            return self
    
    def freeze(self):
        for layer in self.model.layers:
            layer.trainable = False 
        return self
