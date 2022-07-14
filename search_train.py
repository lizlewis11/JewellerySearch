#For BeaconCross solutions Elizabeth Lewis 15/7/2019
#creates feature vectors for entire input dataset of images and stores them in LMDB database
#Index the vectors and store in ann
from Jewelry_Necklace.index_images import ImageIndex
from Jewelry_Necklace import config


#Requirements annoy-1.15.2
#imp liz it is necessary to install numpy-1.16.2, numpy-1.17.0 results in many warnings


index = ImageIndex(config.PATH_FEATURES_LMDB,config.PATH_TREE,config.NO_OF_TREES,config.K_FOR_KNN_SEARCH)
returnVal = index.extract_features_training(config.INPUTCSV,config.BATCH_SIZE)
#returnVal = 0
#if returnVal is 0:#No exceptions then will return 0
    #createIndexTree = NNBuildIndexTree(config.PATH_FEATURES_LMDB,config.PATH_TREE)
    #NO_OF_TREES should be a factor of total number of images
    #createIndexTree.buildTree(config.NO_OF_TREES)

