# Food Ingredients Recognition through Multi-label Learning

This reopsitory contains the code for applying image-based ingredients recognition through multi-label learning.

## Usage

* Modify the file train.sh including the paths to multimodal_keras_wrapper library and to Keras.
* Prepare the dataset following the same format as multimodal_keras_wrapper/data/sample_data.
* Insert the data paths in config.py and modify any parameter as desired.
* Run ./train.sh for training a model.

## Dependencies

* [Multimodal Keras Wrapper](https://github.com/MarcBS/multimodal_keras_wrapper/releases/tag/0.7) >= 0.7
* [Custom Keras fork](https://github.com/MarcBS/keras/releases) >= 1.2.3

 Datasets

The datasets Ingredients101 and Recipes5k were used to evaluate this model:
* Ingredients 101:  It consists of the list of most common ingredients for each of the 101 types of food contained in the Food101 dataset, making a total of 446 unique ingredients (9 per recipe on average). The dataset was divided in training, validation and test splits making sure that the 101 food types were balanced. We make public the lists of ingredients together with the train/val/test split applied to the images from the Food101 dataset.
* Recipes5k: dataset for ingredients recognition with 4,826 unique recipes composed of an image and the corresponding list of ingredients. It contains a total of 3,213 unique ingredients (10 per recipe on average). Each recipe is an alternative way to prepare one of the 101 food types in Food101. Hence, it captures at the same time the intra-class variability and inter-class similarity of cooking recipes. The nearly 50 alternative recipes belonging to each of the 101 classes were divided in train, val and test splits in a balanced way. We make also public this dataset together with the splits division. A problem when dealing with the 3,213 raw ingredients is that many of them are sub-classes (e.g. ’sliced tomato’ or ’tomato sauce’) of more general versions of themselves (e.g. ’tomato’). Thus, we propose a simplified version by applying a simple removal of overlydescriptive particles (e.g. ’sliced’ or ’sauce’), resulting in 1,013 ingredients used for additional evaluation
