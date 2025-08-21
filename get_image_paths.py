import os
from glob import glob

def get_image_paths(data_path, categories, num_train_per_cat):
    num_categories = len(categories)

    train_image_paths = []
    test_image_paths = []
    val_image_paths = []

    train_labels = []
    test_labels = []
    val_labels = []

    for category in categories:
        image_paths = glob(os.path.join(data_path, 'train', category, '*.tif'))
        for i in range(num_train_per_cat):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)

        image_paths = glob(os.path.join(data_path, 'test', category, '*.tif'))
        for i in range(len(image_paths)):
            test_image_paths.append(image_paths[i])
            test_labels.append(category)
        
        image_paths = glob(os.path.join(data_path, 'val', category, '*.tif'))
        for i in range(len(image_paths)):
            val_image_paths.append(image_paths[i])
            val_labels.append(category)

    return train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels
