"""
load_data.py
------------
Script testing image data loading using tf.data (with tensorflow 2.1!
Note that tensorflow should thus be updated to 2.1 (there is already an environment)

Based on: https://www.tensorflow.org/tutorials/load_data/images

As stated in that documentation, other option is keras.preprocessing, but its performance is worse

More information about tf.data for pipeline performance optimization: https://www.tensorflow.org/guide/data_performance
"""


import tensorflow as tf

# processing of multiple images in parallel with optimum number of parallel calls
AUTOTUNE = tf.data.experimental.AUTOTUNE

import pandas as pd


# Check if tensorflow version is 2.1.0
print(tf.__version__)

# Test if gpu support is available
print(tf.config.list_physical_devices('GPU'))   # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
print(tf.test.is_built_with_cuda())     # True if GPU CUDA support


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# NOTE: Functions from docs are implemented to work with the following file structure:
#       data
#       |______ label1: [l1_img_1, l1_img_2, ...]
#       |______ labelN: [l1_img_1, l2_img_2, ...]
# 
# However, current structure is:
#       Images
#       |------ img1, img2, ...
# And in a separate excel file, labels are noted as:
#       |------ labels.xls: (img_number, label)
# 
# Either way, the OBJETIVE is to return, with an implemented process_path() function, 
# a TUPLE with the decoded image as a tensor, and its label: (IMAGE, LABEL)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# :::: Parameters for the loader :::::::::::::::::::::::::::::::::::::::::::::::::::::::::

IMG_HEIGHT, IMG_WIDTH = 1288, 1288  # pixels

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# ::::: DATA LOADING FUNCTIONS: Pure Tensorflow functions -> performance :::::::::::::::::

# TODO: Implement in order to get labels from xls tf.data.csv
def get_label(labels_path, n_images=None):
    """ Retrieve a dataframe with the Image ID and its corresponding label """
    
    # Read xls file:
    read_data = pd.read_excel('labels.xlsx')
    labels_df = read_data[['Classification', 'idImage']]

    labels_df = labels_df.dropna(how = 'any') # drop images withot classification
    labels_df['Classification'] = labels_df['Classification'].apply(lambda x: int(x)) # convert labels to ints
    labels_df = labels_df.loc[(labels_df['Classification'] >= 0) & (labels_df['Classification'] < 3)] # remove defective images (label = -1) and cat 3

    if n_images is not None:
        labels_df = labels_df.loc[labels_df['Image_Name'] <= n_images] # get the selected number of images

    return labels_df


def decode_image(img):
    
    # Conver the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Conver to floats in a [0, 1] scaled range (same effect as typical ./255)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # resize image to desired size
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    
    # Load label from the name of the image (number)
    # print(str(file_path))
    labels_df = get_label("labels.xlsx")
    # NOTE: This is a dataframe with all albel and images!
    
    # Load the raw data from the file as string
    img = tf.io.read_file(file_path)
    
    # TODO: FROM HERE: HOW TO DETECT THE IMAGE NAME? File path is a tensor!
    print(f"File path: {file_path.list_files.take(1).numpy()}")
    
    label = labels_df[labels_df["idImage"] == file_path]   #TODO: IS FILEPATH THE IMAGE ID?
    
    # decode the 'jpeg' image using the implemented function above
    img = decode_image(img)
    
    # return the pair of 'decoded image' & 'label'
    return img, label

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Save files from Images data folder. This is a tf.data.Dataset with the strings of all images name
list_ds = tf.data.Dataset.list_files("Images/*")

# Print some of the datasets files
for f in list_ds.take(5):
    print(f.numpy())

# Parallelize data loading using AUTOTNE and using map to create dataset of image, label pairs
# NOTE: Since list_ds is already a ts.dataset, it doesn't need to especify the filepath argument
print("\nProcessing path images ...")
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# TODO:
# - Pass excel as arguments
# - Transform labels data frame to a tensorflow dataset, in order to perform the 
#   image id and label mapping with better performance.
# - Program a script to transfrom file structure to data/{label1, label2, ..}/{1, 2, 3, ...}