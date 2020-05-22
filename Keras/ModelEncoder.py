from json import JSONEncoder
import json
import ExplainerClassifierCNN
from Explainer import Explainer
import VGG
from ResNet50Mod import ResNet50ModClf
from keras import Model
class ModelEncoder(JSONEncoder):

    def default(self, object):

        if (isinstance(object, ExplainerClassifierCNN.ExplainerClassifierCNN) or 
            isinstance(object, Explainer) or
            isinstance(object, VGG.VGGClf) or
            isinstance(object, ResNet50ModClf)
            ):

            return object.__dict__
        
