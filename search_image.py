#For BeaconCross solutions Elizabeth Lewis 15/7/2019
#takes query image, extracts feature vector and compares distance based on index file
from Jewelry_Necklace import config
from Jewelry_Necklace.index_images import ImageIndex
import cv2
from PIL import Image


query_img_path = config.QUERY_IMG
query_img = cv2.imread(query_img_path)
if query_img is not None:
    cv2.imshow("Query",query_img)
    cv2.waitKey(0)
    # send query image for search
    index = ImageIndex(config.PATH_FEATURES_LMDB, config.PATH_TREE, config.NO_OF_TREES,config.K_FOR_KNN_SEARCH)
    ret_val = index.initialize()

    if ret_val is 0:

        str_value = index.search_image(query_img_path, config.NO_SAME_SIMILAR_IMAGES)

        # imagepaths of images returned by search algorithm
        if str_value is not None:
            img_counter = 1
            for val in str_value:
                pos = val.find(',')
                image_path = val[:pos]
                sku_id = val[pos + 1:]
                print(sku_id)

                result_img = cv2.imread(image_path)
                pos = image_path.rfind('\\') + 1
                img_id = image_path[pos:]
                cv2.imshow("Result_{} id:{}".format(img_counter,img_id), result_img)
                img_counter = img_counter + 1
                cv2.waitKey(0)

                # write csv

    else:
        print("Error initializing db and index tree")
else:
    print("JewellerySearch_Transfer: Query image is None, Pls check image path")




