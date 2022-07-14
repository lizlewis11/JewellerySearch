# JewellerySearch
This code allows you to search though your database and find similar Jewellery for recommendations on your ecomm store. The search is optimized and very fast.

During training, create feature vectors for entire input dataset of images and stores them in LMDB database. The training is conducted at midnight during non-operational hours.
Index the vectors and store.

At the time of search, query image goes through the same pre-processing and the feature vector is compared with Index of vectors to find vector with minimum distance or distance less than the given threshold.

This enables fast recommendation search even on mobile phones.
