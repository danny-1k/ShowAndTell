if [ -d "../data/coco" ]
then
    echo "Coco dataset already downloaded..."
else
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ../data/coco
    wget http://images.cocodataset.org/zips/train2014.zip -P ../data/coco
    wget http://images.cocodataset.org/zips/val2014.zip -P ../data/coco

    unzip ../data/coco/train2014.zip -d ../data/coco
    rm ../data/coco/train2014.zip 

    unzip ../data/coco/val2014.zip -d ../data/coco
    rm ../data/coco/val2014.zip 

    unzip ../data/coco/annotations_trainval2014.zip -d ../data/coco
    rm ../data/coco/annotations_trainval2014.zip
fi