from tensorflow.python.util.tf_export import tf_export
from tensorflow.keras.callbacks import Callback
import numpy as np
from keras.models import Model

@tf_export("keras.callbacks.ModelCheckpoint")
class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, decmaker_filepath, exp_filepath, base_model, decmaker, explainer, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.decmaker_filepath = decmaker_filepath
        self.exp_filepath = exp_filepath
        self.base_model = base_model
        self.decmaker = decmaker
        self.explainer = explainer
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            decmaker_filepath = self.decmaker_filepath.format(epoch=epoch + 1, **logs)
            exp_filepath = self.exp_filepath.format(epoch=epoch + 1, **logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            #trainable_config = []
                            #print("Setting all layers to trainable")
                            #for layer in self.base_model.layers:
                                #print(layer, layer.trainable)
                                #trainable_config.append(layer.trainable)
                                #layer.trainable = True
                                #if(isinstance(layer, Model)):
                                    #for sublayer in layer.layers:
                                        #print(sublayer, sublayer.trainable)
                                        #trainable_config.append(sublayer.trainable)
                                        #sublayer.trainable = True
                                #print(" ")
                            self.base_model.save_weights(filepath, overwrite=True)
                            self.decmaker.save_weights(decmaker_filepath, overwrite=True)
                            self.explainer.save_weights(exp_filepath, overwrite=True)
                            #self.model.save_weights(full_filepath, overwrite=True)
                        
                            #print("Reverting previous operation")
                            #i = 0
                            #for layer in self.base_model.layers:
                                    #print("Before", layer, layer.trainable)
                                    #layer.trainable = trainable_config[i]
                                    #i += 1
                                    #print("After", layer, layer.trainable)
                                    #if(isinstance(layer, Model)):
                                        #for sublayer in layer.layers:
                                            #print("Before", sublayer, sublayer.trainable)
                                            #sublayer.trainable = trainable_config[i]
                                            #i += 1
                                            #print("After", sublayer, sublayer.trainable)
                                    #print(" ")
                        
                        else:
                            trainable_config = []
                            print("Setting all layers to trainable")
                            for layer in self.base_model.layers:
                                #print(layer, layer.trainable)
                                trainable_config.append(layer.trainable)
                                layer.trainable = True
                                if(isinstance(layer, Model)):
                                    for sublayer in layer.layers:
                                        #print(sublayer, sublayer.trainable)
                                        trainable_config.append(sublayer.trainable)
                                        sublayer.trainable = True
                                #print(" ")

                            self.base_model.save(filepath, overwrite=True)
                            self.decmaker.save(decmaker_filepath, overwrite=True)
                            self.explainer.save(exp_filepath, overwrite=True)
                            #self.model.save(full_filepath, overwrite=True)

                            print("Reverting previous operation")
                            i = 0
                            for layer in self.base_model.layers:
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

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    #trainable_config = []
                    #print("Setting all layers to trainable")
                    #for layer in self.base_model.layers:
                       # print(layer, layer.trainable)
                        #trainable_config.append(layer.trainable)
                        #layer.trainable = True

                    self.base_model.save_weights(filepath, overwrite=True)
                    self.decmaker.save_weights(decmaker_filepath, overwrite=True)
                    self.explainer.save_weights(exp_filepath, overwrite=True)
                    #self.model.save_weights(full_filepath, overwrite=True)

                    #print("Reverting previous operation")
                    #i = 0
                    #for layer in self.base_model.layers:
                            #print(layer, layer.trainable)
                           # layer.trainable = trainable_config[i]
                            #i += 1
                else:
                    trainable_config = []
                    print("Setting all layers to trainable")
                    for layer in self.base_model.layers:
                        #print(layer, layer.trainable)
                        trainable_config.append(layer.trainable)
                        layer.trainable = True
                        if(isinstance(layer, Model)):
                            for sublayer in layer.layers:
                                #print(sublayer, sublayer.trainable)
                                trainable_config.append(sublayer.trainable)
                                sublayer.trainable = True
                        #print(" ")

                    self.base_model.save(filepath, overwrite=True)
                    self.decmaker.save(decmaker_filepath, overwrite=True)
                    self.explainer.save(exp_filepath, overwrite=True)
                    #self.model.save(full_filepath, overwrite=True)

                    print("Reverting previous operation")
                    i = 0
                    for layer in self.base_model.layers:
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
