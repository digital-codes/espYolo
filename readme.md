# TFLITE object detection on ESP32S3

## Model

### Get model info

```
import tensorflow as tf
modelDest="final_model_regions_wide.tflite"
interpreter = tf.lite.Interpreter(model_path=modelDest)
interpreter.allocate_tensors()
print("Input:", interpreter.get_input_details())
print("Output:", interpreter.get_output_details())
```

> Input: [{'name': 'serving_default_input_layer:0', 'index': 0, 'shape': array([  1, 160, 160,   3], dtype=int32), 'shape_signature': array([ -1, 160, 160,   3], dtype=int32), 'dtype': <class 'numpy.int8'>, 'quantization': (0.003921568859368563, -128), 'quantization_parameters': {'scales': array([0.00392157], dtype=float32), 'zero_points': array([-128], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

> Output: [{'name': 'StatefulPartitionedCall_1:0', 'index': 25, 'shape': array([   1, 1125], dtype=int32), 'shape_signature': array([  -1, 1125], dtype=int32), 'dtype': <class 'numpy.int8'>, 'quantization': (0.00390625, -128), 'quantization_parameters': {'scales': array([0.00390625], dtype=float32), 'zero_points': array([-128], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]



## Images 

### External Datasets

https://www.kaggle.com/code/juniorbueno/detecting-objects-with-yolo
=> git clone https://github.com/AlexeyAB/darknet


https://storage.googleapis.com/openimages/web/download_v7.html#download-manually

 Download Manually
Images
If you're interested in downloading the full set of training, test, or validation images (1.7M, 125k, and 42k, respectively; annotated with bounding boxes, etc.), you can download them packaged in various compressed files from CVDF's site:

If you only need a certain subset of these images and you'd rather avoid downloading the full 1.9M images, we provide a Python script that downloads images from CVDF.

    Download the file downloader.py (open and press Ctrl + S), or directly run:

    wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py

    Create a text file containing all the image IDs that you're interested in downloading. It can come from filtering the annotations with certain classes, those annotated with a certain type of annotations (e.g., MIAP). Each line should follow the format $SPLIT/$IMAGE_ID, where $SPLIT is either "train", "test", "validation", or "challenge2018"; and $IMAGE_ID is the image ID that uniquely identifies the image. A sample file could be:

    train/f9e0434389a1d4dd
    train/1a007563ebc18664
    test/ea8bfd4e765304db

    Run the following script, making sure you have the dependencies installed:

    python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER --num_processes=5

    For help, run:

    python downloader.py -h

Annotations and metadata
Image IDs
Image labels
Boxes
Segmentations
Relationships
Localized narratives
Localized narratives voice recordings
Point-labels
Metadata

### traffic sign
https://www.kaggle.com/datasets/pkdarabi/cardetection


### 26 objects
https://www.kaggle.com/datasets/mohamedgobara/26-class-object-detection-dataset

### road signs

https://www.kaggle.com/datasets/fhabibimoghaddam/road-sign-recognition

### street objects
https://www.kaggle.com/datasets/owm4096/street-objects


### Synthetic
imggen 


## ESP deployment

use idf version >=5, e.g. 5.5 via 

> source /opt/esp32/repo2/esp-idf/export.sh

set target **!! overwrites config. re-run menuconfig after**

> idf.py set-target esp32s3

menuconfig

  * psram is octal mode, aut detect (psram settings)
  * flash size is 8MB (serial flasher)
  * cpu speed is 240MHz (system settings)
  * cache settings to max (system settings)
  * camera => GC0308 (camera settings)

optionally fullclean

> idf.py fullclean

compile 

> idf.py build

or 

> IDF_TARGET="esp32s3" idf.py build



flash 

> idf.py -p <port> flash

monitor **!! Terminate with CTRL-]  or CTRL-T + CTRL-X**

> idf.py -p <port> monitor 

or both 

> idf.py -p <port> flash monitor 

### Example

using *robot_204* image on region grid 5*5, 160x160 image size

expected results: 9 objects

results:
> Output vector len: 1125
  Activation at 6: 0.500000
  Activation at 43: 0.500000
  Activation at 109: 0.500000
  Activation at 707: 0.500000
  Activation at 723: 0.500000
  Activation at 725: 0.500000
  Activation at 778: 0.500000
  Activation at 805: 0.500000

results must be mapped to regions

