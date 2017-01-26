import logging
import numpy as np
from timeit import default_timer as timer
import copy

from keras_wrapper.cnn_model import saveModel, loadModel
from keras_wrapper.extra.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
from keras_wrapper.extra.read_write import *

from config import load_parameters
from data_engine.prepare_data import build_dataset
from model import Ingredients_Model

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
        Main function
    """

    if(params['RELOAD'] > 0):
        logging.info('Resuming training.')
        
    ########### Load data
    dataset = build_dataset(params)
    ###########

    ########### Build model
    if(params['RELOAD'] == 0): # build new model
        ing_model = Ingredients_Model(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                                model_name=params['MODEL_NAME'], store_path=params['STORE_PATH'])

        # Define the inputs and outputs mapping from our Dataset instance to our model
        ing_model.setInputsMapping(params['INPUTS_MAPPING'])
        ing_model.setOutputsMapping(params['OUTPUTS_MAPPING'])

    else: # resume from previously trained model
        ing_model = loadModel(params['STORE_PATH'], params['RELOAD'])
        ing_model.setOptimizer()
    ###########
    
    
    ########### Callbacks
    callbacks = buildCallbacks(params, ing_model, dataset)
    ###########
    

    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'], 'batch_size': params['BATCH_SIZE'],
                       'lr_decay': params['LR_DECAY'], 'lr_gamma': params['LR_GAMMA'],
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'], 'epoch_offset': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       'patience': params['PATIENCE'], 'metric_check': params['STOP_METRIC']
                       }
    ing_model.trainNet(dataset, training_params)
    
    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('Total time spent {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
    ###########
    

def apply_model(params):
    """
        Function for using a previously trained model for predicting.
    """
    
    
    ########### Load data
    dataset = build_dataset(params)
    ###########
    
    
    ########### Load model
    ing_model = loadModel(params['STORE_PATH'], params['RELOAD'])
    ing_model.setOptimizer()
    ###########
    

    ########### Apply sampling
    for s in params["EVAL_ON_SETS"]:

        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s], 'normalize_images': params['NORMALIZE_IMAGES'], 
                             'mean_substraction': params['MEAN_SUBSTRACTION']}
        predictions = ing_model.predictNet(dataset, params_prediction)[s]
        
        # Format predictions
        if params['TRAINING_STAGE'] == 'CAM':
            predictions = predictions[0]
        predictions = ing_model.decode_predictions(predictions, 1, # not used
                                             dataset.extra_variables['idx2word_binary'])
        
        # Store result
        filepath = ing_model.model_path+'/'+ s +'_labels.pred' # results file
        listoflists2file(filepath, predictions)
    ###########

    
def buildCallbacks(params, model, dataset):
    """
        Builds the selected set of callbacks run during the training of the model
    """
    
    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = dict()
        extra_vars['n_parallel_loaders'] = params['PARALLEL_LOADERS']

        if params['CLASSIFICATION_TYPE'] == 'single-label':

            extra_vars['n_classes'] = params['NUM_CLASSES']
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                exec ("extra_vars[s]['references'] = dataset.Y_" + s)
            vocab = None
            is_text = False

        elif params['CLASSIFICATION_TYPE'] == 'multi-label':
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['word2idx'] = dataset.extra_variables['word2idx_binary']
                exec("extra_vars[s]['references'] = dataset.Y_"+s+"[params['OUTPUTS_IDS_DATASET'][0]]")
            vocab = dataset.extra_variables['idx2word_binary']
            is_text = True

        callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model, dataset, gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                   metric_name=params['METRICS'], set_name=params['EVAL_ON_SETS'],
                                                                   batch_size=params['BATCH_SIZE'],
                                                                   is_text=is_text, index2word_y=vocab, # text info
                                                                   save_path=model.model_path,
                                                                   reload_epoch=params['RELOAD'],
                                                                   start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                                                   write_samples=params['WRITE_VALID_SAMPLES'],
                                                                   write_type='listoflists',
                                                                   extra_vars=extra_vars,
                                                                   verbose=params['VERBOSE'])
        callbacks.append(callback_metric)

    
    
    return callbacks


if __name__ == "__main__":
 
    params = load_parameters()
    if(params['MODE'] == 'training'):
        logging.info('Running training.')
        train_model(params)
    elif(params['MODE'] == 'predict'):
        logging.info('Running predict.')
        apply_model(params)

    logging.info('Done!')   



