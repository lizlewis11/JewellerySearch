from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import lmdb
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
import logging

#For BeaconCross solutions Elizabeth Lewis 15/7/2019
#Class FeatureExtracter
#This class is used to pass all the images through VGG16 model layers except the final softmax layer
#We extract all the features from the last layer which are vectors, this is done by function ExtractFeaturesSet

#input search/query image is also subjected to all preprocessing exactly as input image feature set
#and then passed through VGG16 layers to get vector, this is done by function Extract

#the distance between query vector and all feature vectors are then calculated using KNN.py

logger = logging.getLogger(__name__)


class ImageIndex:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __init__(self, db_path, tree_path, no_of_trees, k_KNN):
        # initialize VGG16 model with imagenet weights
        self.model = VGG16(weights="imagenet",include_top = False)
        # for multi threading in flask
        global graph
        graph = tf.get_default_graph()

        # 7*7*512(VGG16VectorSize)
        self.vector_size = 25088
        # the size to which the input images are re-sized before passing them through VGG16
        self.img_resize_dim = 224
        self.db_path = db_path
        self.tree_path = tree_path
        self.no_of_trees = no_of_trees
        self.env = None
        self.search_index = None
        # k factor for search
        self.k_KNN = k_KNN
        self.total_images = 0

        # delete existing LMDB if necessary
        #if os.path.exists(lmdb_path) and create:
            #self.logger.debug('Erasing previously created LMDB at %s', lmdb_path)
            #shutil.rmtree(lmdb_path)

    #to extract features of entire image set
    def extract_features_training(self, image_list_csv, batch_size):

        if image_list_csv is "":
            logger.error('CSV with input image paths is not provided')
        else:

            try:
                # take image paths from csv
                input_csv = pd.read_csv(image_list_csv)
                
                # total number of images in image set
                no_of_images = len(input_csv.index)

                image_paths = []
                sku_id = []

                # store all image paths in a list
                for counter in range(0, no_of_images):
                    image_paths.append(input_csv.iloc[counter][0])  # 0 column with image path, first row is column name
                    sku_id.append(input_csv.iloc[counter][1])

                # open a LMDB database and store the resultant image feature vectors
                # key: image_id, integer number 1,2,3,4 (No duplicity, even same image will have different image id)
                # value: image path

                env = lmdb.open(self.db_path, map_size=int(1e9))  # Check if exists os.path.exists
                index = AnnoyIndex(self.vector_size, metric='angular')

                with env.begin(write=True)as txn:  # write into db

                    batch_extra_images = no_of_images % batch_size
                    last_batch_img = no_of_images - batch_extra_images
                    # take batches of images, each batch of size BatchSize, this is done to prevent 'out of memory' err
                    # BatchSize is the step size
                    current_batch_size = batch_size
                    for (batchNo, cnt) in enumerate(range(0, no_of_images, batch_size)):
                        batch_paths = image_paths[cnt:cnt+batch_size]
                        batch_images = []
                        logger.debug('Batch No: %s', batchNo)

                        for img_path in batch_paths:

                            # load image using keras helper utility and resize it to 224x224
                            # not using load_img as its not supporting PNG and transperency images
                            img = self.load_image(img_path)

                            logger.debug('Image properties: mode %s, format %s, size %s', img.mode, img.format, img.size)

                            img = img_to_array(img)

                            # pre-process the image by expanding dim and subtracting mean RGB from ImageNet dataset
                            img = np.expand_dims(img,axis=0)

                            batch_images.append(img)

                        batch_images = np.vstack(batch_images)

                        if cnt >= last_batch_img:  # for last remaining images after batches are over
                            current_batch_size = batch_extra_images
                        # model.predict will give all features for this batch of images
                        features = self.model.predict(batch_images, batch_size=current_batch_size)

                        features = features.reshape((features.shape[0], self.vector_size))#For VGG16

                        for feature_counter in range(0, current_batch_size):  # Num_of_Images
                            print(cnt + feature_counter)
                            index.add_item(cnt + feature_counter, features[feature_counter])
                            str_value = image_paths[cnt + feature_counter] + "," + sku_id[cnt + feature_counter]
                            # important to serialize to bytes before storing in db
                            txn.put(str(cnt + feature_counter).encode(),str(str_value).encode())

                # building the tree
                try:
                    index.build(self.no_of_trees)
                    index.save(self.tree_path)
                    logger.debug('Successfully built index of vectors')
                except Exception as ex:
                    logger.error('Error while building index tree %s', str(ex))

                    logger.debug('Total No. of images processed: %d', no_of_images)
                env.close()
                return 0
                #env.stat() to view environment statistics

            except Exception as e:
                logger.error('Exception occured during training %s', str(e))
                return -1

    # to extract features of query image
    def extract(self, image):

        # load image using keras helper utility and resize it to 224x224
        img = self.load_image(image)
        img = img_to_array(img)

        # pre-process the image by expanding dim and subtracting mean RGB from ImageNet dataset
        img = np.expand_dims(img,axis=0)

        # for multi-threading in flask
        with graph.as_default():
            feature = self.model.predict(img)

        feature = feature.reshape((feature.shape[0], self.vector_size))  # For VGG16

        return feature

    # Handle different image file formats jpg,jpeg,bmp,png
    def load_image(self,image):

        image = Image.open(image)
        # getpalettemode() and mode available only after image load
        image.load()
        if image.mode == 'P':
            # PIL loses palette attribute during crop. So we use low-level api.
            alpha = 'A' in image.im.getpalettemode()
            image = image.convert('RGBA' if alpha else 'RGB')
            logger.debug('Image mode P %s', image.mode)

        # not elif...after P this should be executed
        if image.mode in ('RGBA','LA'):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image,image.split()[-1])
            image = background

        desired_size = self.img_resize_dim
        old_size = image.size  # old_size is in (height, width) format
        if old_size[0] != old_size[1]:  # aspect ratio is not 1
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            resized_blank_image = image.resize(new_size, Image.ANTIALIAS)
            # create a new image and paste the resized on it

            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(resized_blank_image, ((desired_size - new_size[0]) // 2,
                              (desired_size - new_size[1]) // 2))
        else:
            new_im = image.resize((self.img_resize_dim, self.img_resize_dim), Image.ANTIALIAS)

        # image = image.resize((self.img_resize_dim,self.img_resize_dim), Image.ANTIALIAS)
        return new_im

    def initialize(self):

        self.open_db_connection()
        # load the annoy tree
        self.search_index = AnnoyIndex(self.vector_size)
        self.search_index.load(self.tree_path)
        ret_val = 0
        # if db opened successfully
        if self.env is not None:
            self.total_images = self.env.stat()['entries']
            logger.debug('Total images in database %s', self.total_images)

            if self.k_KNN > self.total_images:
                ret_val = -1
                logger.error('Value of K_FOR_KNN_SEARCH should not exceed Total number of images')
        else:
            ret_val = -1
            logger.error('DB connection not open')

        return ret_val

    def open_db_connection(self):

        try:
            self.env = lmdb.open(self.db_path, map_size=int(1e9), readonly=True)
        except Exception as e:
            logger.error('Error while opening database %s', str(e))

    def close_db_connection(self):

        try:
            self.env.close()
        except Exception as e:
            logger.error('Error while closing database %s', str(e))

    def search_image(self, image, num_results):
        try:
            #if num_results <= self.total_images:

            feature_vector = self.extract(image)  # Query 1 image at a time

            nn_results, distances = self.search_index.get_nns_by_vector(feature_vector.flatten(), num_results, search_k=self.k_KNN, include_distances = True)

            image_path_skuid = []
            print(distances)
            with self.env.begin() as txn:

                for count, id in enumerate(nn_results):
                    str_key = str(id).encode()  # same encoding was stored while populating db
                    str_value = txn.get(str_key).decode()  # de-serialise value

                    if distances[count] <= 2:  # show only images with distance less than 80%
                        image_path_skuid.append(str_value)  # get skuid

            return image_path_skuid
            #else:
                #logger.error("Value of NO_SAME_SIMILAR_IMAGES should not exceed Total number of images")

        except Exception as e:
            logger.error('Error occured during image search %s', str(e))
    # Environment.close() (not cursor.close()) should not be called concurrently