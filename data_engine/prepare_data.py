from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

from collections import Counter
from operator import add

import nltk
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

def build_dataset(params):
    
    if params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = params['DATA_ROOT_PATH']
        ds = Dataset(params['DATASET_NAME'], base_path+params.get('SUFFIX_DATASET', '/images'), silence=silence)

        ##### INPUT DATA
        ### IMAGES
        ds.setInput(base_path+'/'+params['IMG_FILES']['train'], 'train',
                   type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                   img_size=params['IMG_SIZE'], img_size_crop=params['IMG_SIZE_CROP'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['val'], 'val',
                   type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                   img_size=params['IMG_SIZE'], img_size_crop=params['IMG_SIZE_CROP'])
        ds.setInput(base_path+'/'+params['IMG_FILES']['test'], 'test',
                   type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                   img_size=params['IMG_SIZE'], img_size_crop=params['IMG_SIZE_CROP'])
        # Set train mean
        ds.setTrainMean(mean_image=params['MEAN_IMAGE'], id=params['INPUTS_IDS_DATASET'][0])

        ##### OUTPUT DATA
        if params['CLASSIFICATION_TYPE'] == 'single-label':

            # train split
            ds.setOutput(base_path + '/' + params['LABELS_FILES']['train'], 'train',
                         type='categorical', id=params['OUTPUTS_IDS_DATASET'][0])
            # val split
            ds.setOutput(base_path + '/' + params['LABELS_FILES']['val'], 'val',
                         type='categorical', id=params['OUTPUTS_IDS_DATASET'][0])
            # test split
            ds.setOutput(base_path + '/' + params['LABELS_FILES']['test'], 'test',
                         type='categorical', id=params['OUTPUTS_IDS_DATASET'][0])

        elif params['CLASSIFICATION_TYPE'] == 'multi-label':

            # Convert list of ingredients into classes
            logging.info('Preprocessing list of ingredients for assigning vocabulary as image classes.')
            [classes, word2idx, idx2word] = convertIngredientsList2BinaryClasses(base_path,
                                                                                 params['LABELS_FILES'],
                                                                                 params['CLASSES_PATH'],
                                                                                 type_list=params.get('LABELS_TYPE_LIST', 'identifiers'))
            # Insert them as outputs
            ds.setOutput(classes['train'], 'train', type='binary', id=params['OUTPUTS_IDS_DATASET'][0])
            ds.setOutput(classes['val'], 'val', type='binary', id=params['OUTPUTS_IDS_DATASET'][0])
            ds.setOutput(classes['test'], 'test', type='binary', id=params['OUTPUTS_IDS_DATASET'][0])

            # Insert vocabularies
            ds.extra_variables['word2idx_binary'] = word2idx
            ds.extra_variables['idx2word_binary'] = idx2word
            
            if 'Food_and_Ingredients' in params['DATASET_NAME']:
                
                # train split
                ds.setOutput(base_path + '/' + params['LABELS_FILES_FOOD']['train'], 'train',
                             type='categorical', id=params['OUTPUTS_IDS_DATASET'][1])
                # val split
                ds.setOutput(base_path + '/' + params['LABELS_FILES_FOOD']['val'], 'val',
                             type='categorical', id=params['OUTPUTS_IDS_DATASET'][1])
                # test split
                ds.setOutput(base_path + '/' + params['LABELS_FILES_FOOD']['test'], 'test',
                             type='categorical', id=params['OUTPUTS_IDS_DATASET'][1])


        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['STORE_PATH'])
    
    
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['STORE_PATH']+'/Dataset_'+params['DATASET_NAME']+'.pkl')

    return ds


def convertIngredientsList2BinaryClasses(base_path, data, multilabels, type_list='identifiers'):

    repeat_imgs = 1
    
    ing_list = []
    counter = Counter()
    with open(base_path+'/'+multilabels) as f:
        for pos_ing, line in enumerate(f):
            # read ingredients
            if type_list == 'identifiers':
                ing = line.rstrip('\n').split(',')
                ing = map(lambda x: x.lower(), ing)
                ing_list.append(ing)
            elif type_list == 'words':
                ing = line.rstrip('\n')
                ing = ing.lower()
                ing_list.append(ing)
                ing = [ing]
            counter.update(ing)

    vocab_count = counter.most_common()

    vocabulary = {}
    list_words = []
    for i, (word, count) in enumerate(vocab_count):
        vocabulary[word] = i
        list_words.append(word)
    len_vocab = len(vocabulary)


    # Preprocess each data split
    classes = dict()
    for set_name, file in data.iteritems():
        classes[set_name] = []
        with open(base_path+'/'+file) as f:
            for idx_img, line in enumerate(f):
                classes[set_name].append(np.zeros((len_vocab,)))
                if type_list=='identifiers':
                    pos_ing = int(line.rstrip('\n'))
                    ings = ing_list[pos_ing]
                elif type_list=='words':
                    ings = line.rstrip('\n').split(',')

                # insert all ingredients
                for w in ings:
                    if w in vocabulary.keys():
                        classes[set_name][-1][vocabulary[w]] = 1

 
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    return [classes, vocabulary, inv_vocabulary]
