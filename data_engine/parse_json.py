import numpy as np
import json
import glob

def parseJson(file):

    # Read file
    file = open(file, 'r')
    data = file.next()
    file.close()
    data = eval(data)

    return data

def getFoodClasses(file_classes, alternative_names):

    classes_list = []
    with open(file_classes, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            line = alternative_names.get(line, line)
            line = ' '.join(line.split('_'))
            classes_list.append(line)

    return classes_list

def getIngredients(data, classes_list, name_field_id='FoodName', ing_field_id='CleanIngredients'):

    classes_json = []
    for food in data:
        food_name = food[name_field_id].lower()
        classes_json.append(food_name)

    ingredients_list = []
    for c in classes_list:
        if c not in classes_json:
            print "WARNING: "+c+" is not in json file."

        ind = classes_json.index(c)
        ingredients_list.append(data[ind][ing_field_id])

    return ingredients_list


def getIngredients_v2(data, imgs_list, name_field_id='RecipeName', ing_field_id='CleanIngredients'):

    # Dictionary from lower to upper case
    classes_dict = dict()
    for folder in data:
        classes_dict[folder.lower()] = folder

    # Find ingredients corresponding to all images
    ingredients_list = []
    final_recipes_list = []
    final_recipes_count = dict()
    for im,folder,idx in imgs_list:
        food_data = data[classes_dict[folder]]

        #assert idx < len(food_data), folder +' '+ str(idx)+' does not exist. len() == '+str(len(food_data))
        if idx >= len(food_data):
            print "WARNING: error, uncomment top line when json is corrected"
            break

        ing_list = food_data[idx][ing_field_id]
        ingredients_list.append(ing_list)

        final_recipe = food_data[idx][name_field_id]

        if final_recipe not in final_recipes_count.keys():
            final_recipes_count[final_recipe] = 1
        else:
            final_recipes_count[final_recipe] += 1
            final_recipe += '_'+str(final_recipes_count[final_recipe])
        final_recipes_list.append(final_recipe)

    return final_recipes_list, ingredients_list


def getIngredients_v3(data, name_field_id='RecipeName', ing_field_id='CleanIngredients', image_field_id='ImagePath'):

    # Find ingredients corresponding to all images
    imgs_list = []
    ingredients_list = []
    final_recipes_list = []
    final_recipes_count = dict()
    for super_class in data:
        food_data = data[super_class]

        idx = 0
        for recipe in food_data:

            # Prepare images list
            im_name = '/'.join(recipe[image_field_id].split('/')[1:])
            imgs_list.append([im_name, super_class, idx])

            # Prepare ingredients list
            ing_list = recipe[ing_field_id]
            ingredients_list.append(ing_list)

            # Prepare recipes list
            final_recipe = recipe[name_field_id]

            if final_recipe not in final_recipes_count.keys():
                final_recipes_count[final_recipe] = 1
            else:
                final_recipes_count[final_recipe] += 1
                final_recipe += '_'+str(final_recipes_count[final_recipe])
            final_recipes_list.append(final_recipe)

            idx += 1


    return final_recipes_list, ingredients_list, imgs_list


def plotStatisticsIngredients(ingredients, plot_whole_list=False):
    all_ingredients = []
    for ing in ingredients:
        ing = [i.lower() for i in ing]
        all_ingredients += ing

    non_repeated = set(all_ingredients)
    non_repeated = sorted(non_repeated)

    if plot_whole_list:
        for ing in non_repeated:
            print ing

    print
    print 'Total number of ingredients: %d' % len(all_ingredients)
    print 'Total number of non-repeated ingredients: %d' % len(non_repeated)

def storeIngredients(ingredients, file):

    with open(file, 'w') as file:
        for ing in ingredients:
            ing = ','.join(ing)
            file.write(ing+'\n')

def storeRecipes(recipes, file):

    with open(file, 'w') as file:
        for r in recipes:
            file.write(r+'\n')

def splitData(imgs, classes, files, split={'train': 0.7, 'val': 0.15, 'test': 0.15}):

    # Split the data in train, val and test
    # Make sure that we have roughly equal proportions for each super class in imgs[:][1] and in classes
    splits_imgs_labels = dict()
    for s in split.keys():
        splits_imgs_labels[s] = []

    # Find all images for each super class
    class_img_dict = dict()
    for i,(im,c,idx) in enumerate(imgs):
        if c not in class_img_dict.keys():
            class_img_dict[c] = []
        class_img_dict[c].append([im, i])

    # Divide the images from each super class on all splits
    for c,v in class_img_dict.iteritems():
        picked_so_far = 0
        # generate data splits
        available_imgs = len(v)
        randomized = np.random.choice(range(len(v)), available_imgs, replace=False)
        randomized = np.array([v[i] for i in randomized])

        for s,p in split.iteritems():
            last_picked = np.ceil(picked_so_far+available_imgs*p)
            splits_imgs_labels[s] += randomized[picked_so_far:last_picked]
            picked_so_far = last_picked

    # Store result in files
    for s, f in files.iteritems():
        f_im = open(f[0], 'w')
        f_lab = open(f[1], 'w')

        for im_path, label in splits_imgs_labels[s]:
            f_im.write(im_path+'\n')
            f_lab.write(label+'\n')

        f_im.close()
        f_lab.close()


def parseImgsRecipes(classes, imgs_folder, img_format='jpg'):
    imgs_list = []
    for c in classes:
        c_ = '_'.join(c.split())
        images = glob.glob(imgs_folder+'/'+c_+'/*'+img_format)

        # Sort images
        im_ids = []
        for im in images:
            imid = im.split('/')[-1].split('_')[0]
            im_ids.append(int(imid))
        sorted_ids = [i[0] for i in sorted(enumerate(im_ids), key=lambda x:x[1])]
        images = [images[i] for i in sorted_ids]

        # Store in complete list
        for i,im in enumerate(images):
            im = '/'.join(im.split('/')[2:])
            imgs_list.append([im, c, i])

    return imgs_list

######################################################################
######################################################################

base_path_food101 = '/media/HDD_3TB/DATASETS/Ingredients101/Annotations'

alternative_names = {'cup_cakes':'cupcakes', 'ice_cream':'chocolate_ice_cream', 'tacos':'beef_tacos'}

# Food101 processing

data = parseJson(base_path_food101+'/'+'recipesData_v2.json')
classes = getFoodClasses(base_path_food101+'/'+'classes.txt', alternative_names)
ingredients_list = getIngredients(data, classes)
plotStatisticsIngredients(ingredients_list)
#storeIngredients(ingredients_list, base_path_food101+'/'+'ingredients.txt')


"""

base_path_recipes5k = 'Recipes5k'

# Recipes 5k processing

classes = getFoodClasses(base_path_recipes5k+'/'+'classes_Food101.txt', alternative_names)
#imgs = parseImgsRecipes(classes, base_path_recipes5k+'/images')
data = parseJson(base_path_recipes5k+'/'+'recipesWideData.json')
#recipes, ingredients_list = getIngredients_v2(data, imgs, name_field_id='RecipeName')
recipes, ingredients_list, imgs = getIngredients_v3(data, name_field_id='RecipeName', image_field_id='ImagePath')
plotStatisticsIngredients(ingredients_list)
storeIngredients(ingredients_list, base_path_recipes5k+'/'+'ingredients_Recipes5k.txt')
storeRecipes(recipes, base_path_recipes5k+'/'+'classes_Recipes5k.txt')
splitData(imgs, classes, files={'train': [base_path_recipes5k+'/'+'train_images.txt',
                                          base_path_recipes5k+'/'+'train_labels.txt'],
                                'val': [base_path_recipes5k+'/'+'val_images.txt',
                                        base_path_recipes5k+'/'+'val_labels.txt'],
                                'test': [base_path_recipes5k+'/'+'test_images.txt',
                                        base_path_recipes5k+'/'+'test_labels.txt']})
"""

print 'Done!'
