#Configuration for BeaconCross fast optimized Image Search using Deep Learning

#Input CSV with path of all images in disk
INPUTCSV = "C:\\Users\\wielj\\Work\\VisualSearch\\Jewellery\\jewelcsv.csv"

#For a large number of images, to avoid memory getting full we execute images in batches
#specify batch size here
BATCH_SIZE = 32 #advised to keep it >= 32

#Number of trees in search index
NO_OF_TREES = 20

#path of LMDB database
PATH_FEATURES_LMDB = "C:\\Users\\wielj\\Work\\VisualSearch\\Jewellery\\JewelleryFeatures.lmdb"

#path of search index
PATH_TREE = "C:\\Users\\wielj\\Work\\VisualSearch\\Jewellery\\JewelleryImageSearch.ann"
#J200000284
#image which has to be searched or need recommendation
QUERY_IMG = "C:\\Users\\wielj\\Work\\imagefetcher\\WebScraper\\Jewellery\\Resize2\\J300000117.jpg"
#8ec54d662f86456abea3e0aa49aa2666.JPG
#buckle\\view_1\\8aaffa6f4f5a4b4da1af952fc1a5d50e.JPG

#number of images in search results
NO_SAME_SIMILAR_IMAGES = 5 #NO_SAME_SIMILAR_IMAGES should not exceed Total number of images.

#K factor for search
K_FOR_KNN_SEARCH = 64 #K_FOR_KNN_SEARCH should not exceed Total number of images.



#Known issue: if number of images less than previous then persists in db...hv to delete db