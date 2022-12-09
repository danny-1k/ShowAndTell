# Set shit up
mkdir ../data
# download dataset. Make sure to have kaggle.json file in ~/.kaggle
kaggle datasets download "hsankesara/flickr-image-dataset" &&\
unzip flickr-image-dataset.zip &&\
mv flickr30k_images flickr30k &&\
mv -r flickr30k ShowAndTell/data/flickr30k
# Fix the dataset
cd ../src && python fix_flickr_dataset.py
