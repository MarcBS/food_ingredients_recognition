from keras.engine import Input
from keras.engine.topology import merge
from keras.layers.core import Dropout, RepeatVector, Merge, Dense, Flatten, Activation, TimeDistributedDense, Lambda
from keras.layers.embeddings import Embedding
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, ClassActivationMapping
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential, Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

from keras_wrapper.cnn_model import Model_Wrapper

import numpy as np
import cPickle as pk
import os
import logging
import shutil
import time
import copy


class Ingredients_Model(Model_Wrapper):
    
    def __init__(self, params, type='VGG16', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, store_path=None, seq_to_functional=False):
        """
            Ingredients_Model object constructor.
            
            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param store_path: path to the folder where the temporal model packups will be stored
            :param seq_to_functional: defines if we are loading a set of weights from a Sequential model to a FunctionalAPI model (only applicable if weights_path is not None)

        """
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)
        
        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']
        
        self.verbose = verbose
        self._model_type = type
        self.params = params

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path)

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" Ingredients_Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('Ingredients_Model type "'+ type +'" is not implemented.')
        
        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path, seq_to_functional=seq_to_functional)
        
        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()
        
        
    def setOptimizer(self, metrics=['acc']):

        """
            Sets a new optimizer for the model.
        """

        super(self.__class__, self).setOptimizer(lr=self.params['LR'],
                                                 loss=self.params['LOSS'],
                                                 optimizer=self.params['OPTIMIZER'],
                                                 loss_weights=self.params.get('LOSS_WIGHTS', None),
                                                 sample_weight_mode='temporal' if self.params.get('SAMPLE_WEIGHTS', False) else None)

    
    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the model instance.
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path


        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    def decode_predictions(self, preds, temperature, index2word, sampling_type=None, verbose=0):
        """
        Decodes predictions

        In:
            preds - predictions codified as the output of a softmax activiation function
            temperature - temperature for sampling (not used for this model)
            index2word - mapping from word indices into word characters
            sampling_type - sampling type (not used for this model)
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        labels_pred = np.where(np.array(preds) > 0.5, 1, 0)
        labels_pred = [[index2word[e] for e in np.where(labels_pred[i] == 1)[0]] for i in range(labels_pred.shape[0])]
        return labels_pred
    
    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for visualization
    # ------------------------------------------------------- #
    
    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'
        
        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'
            
        obj_str += '\n'
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'
            
        obj_str += '-----------------------------------------------------------------------------------'
        
        return obj_str
    
    
    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def Arch_D(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        activation_type_food = params['CLASSIFIER_ACTIVATION_FOOD']
        
        nOutput = params['NUM_CLASSES']
        nOutput_food = params['NUM_CLASSES_FOOD']

        ##################################################
        # Load VGG16 model pre-trained on ImageNet
        self.model = VGG16(weights='imagenet',
                            layers_lr=params['PRE_TRAINED_LR_MULTIPLIER'],
                            input_name=self.ids_inputs[0])

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        fc1_out = self.model.get_layer('fc1').output
        ##################################################

        # Ingredients classification path
        fc_ing = Dense(1024, name='fc_ing',
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(fc1_out)
        ing_out = Dense(nOutput, activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(fc_ing)
        
        
        # Food classification path
        fc_food = Dense(4096, name='fc_food',
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(fc1_out)
        food_out = Dense(nOutput_food, activation=activation_type_food, name=self.ids_outputs[1],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(fc_food)

        self.model = Model(input=image, output=[ing_out, food_out])
    
    
    def VGG16(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        nOutput = params['NUM_CLASSES']

        ##################################################
        # Load VGG16 model pre-trained on ImageNet
        self.model = VGG16(weights='imagenet',
                            layers_lr=params['PRE_TRAINED_LR_MULTIPLIER'],
                            input_name=self.ids_inputs[0])

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('fc2').output
        ##################################################

        # Create last layer (classification)
        x = Dense(nOutput, activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(x)

        self.model = Model(input=image, output=x)


    def Inception(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        nOutput = params['NUM_CLASSES']

        ##################################################
        # Load VGG16 model pre-trained on ImageNet
        self.model = InceptionV3(weights='imagenet')

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('flatten').output
        ##################################################

        # Create last layer (classification)
        x = Dense(nOutput, activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(x)

        self.model = Model(input=image, output=x)

    def ResNet50(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        nOutput = params['NUM_CLASSES']

        ##################################################
        # Load ResNet50 model pre-trained on ImageNet
        self.model = ResNet50(weights='imagenet',
                              layers_lr=params['PRE_TRAINED_LR_MULTIPLIER'],
                              input_shape=tuple([params['IMG_SIZE_CROP'][2]] + params['IMG_SIZE_CROP'][:2]),
                              include_top=False, input_name=self.ids_inputs[0])

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'avg_pool'
        x = self.model.get_layer('avg_pool').output
        ##################################################

        # Create last layer (classification)
        x = Flatten()(x)
        x = Dense(nOutput, activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(x)

        self.model = Model(input=image, output=x)

        
    def TestModel(self, params):
        
        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        nOutput = params['NUM_CLASSES']

        ##################################################
        # Load ResNet50 model pre-trained on ImageNet
        self.model = ResNet50(weights='imagenet',
                              layers_lr=params['PRE_TRAINED_LR_MULTIPLIER'],
                              input_shape=tuple([params['IMG_SIZE_CROP'][2]] + params['IMG_SIZE_CROP'][:2]),
                              include_top=False, input_name=self.ids_inputs[0])

        # Recover input layer
        image = self.model.get_layer(self.ids_inputs[0]).output
        ##################################################

        # Create last layer (classification)
        x = Flatten()(image)
        x = Dense(nOutput, activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(x)

        self.model = Model(input=image, output=x)

        
    # Auxiliary functions
    def changeClassifier(self, params, last_layer='flatten'):
        
        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        activation_type = params['CLASSIFIER_ACTIVATION']
        
        ########
        inp = self.model.get_layer(self.ids_inputs[0]).output
        
        last = self.model.get_layer(last_layer).output
        
        out = Dense(params['NUM_CLASSES'], activation=activation_type, name=self.ids_outputs[0],
                  W_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'],
                  b_learning_rate_multiplier=params['NEW_LAST_LR_MULTIPLIER'])(last)
        
        self.model = Model(input=inp, output=out)
        