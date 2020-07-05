# ClimbingHoldsRecognition
<p>
    The projet will use Deep Learning to segment climbing holds.
    To do, we are using
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras">Keras</a>
    and 
    <a href=https://github.com/tensorflow/tensorflow>Tensorflow</a>
    as library in Python.
</p>
<p>
    We are using
    <a href=https://github.com/jsbroks/coco-annotator>Coco-annotator</a>
    to annotate climbing holds on pictures.
</p>
<p>
    Moreover the major part of our source code come from
    <a href=https://github.com/matterport/Mask_RCNN>matterport/Mask_RCNN</a>
    and specially from
    <a href=https://https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon>
        Mask_RCNN_Balloon
    </a>
</p>

## Install
I highly recommend you to install in a virtual environment (such as in ***./init-virtualenv.sh***)

Download pre-train weight from [Mask_RCNN_Balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
in the root folder of the project with:
```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```

### Build
First, build coco lib (for coco dataset)
```
cd coco-master/PythonAPI
pip3 install Cython
pip3 install numpy
make
make install
```

Then build project
```
cd ../..
pip3 install -r requirements.txt
python3 setup.py install
```

## Uses with Command Line
### Train
```
python main.py train --dataset=./datasets/paffme --weights=trained --logs=./logs/paffme
```
'***--weights***' can be '***.h5***' file (from previous train)
### Detection
Visualise results
```
python detection.py visualize --weights=./path/to/weight.h5 --image=./path/to/img.jpg
```
Export results (just bounding box currently)
```
python detection.py export --weights=./path/to/weight.h5 --image=./path/to/img.jpg --file_save=./path/to/export/file.json
```
Both of previous functionality
```
python detection.py visualize_and_export --weights=./path/to/weight.h5 --image=./path/to/img.jpg --file_save=./path/to/export/file.json
```