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

flash 

> idf.py -p <port> flash

monitor **!! Terminate with CTRL-]  or CTRL-T + CTRL-X**

> idf.py -p <port> monitor 

or both 

> idf.py -p <port> flash monitor 

