# Recogonize Flowers with TensorFLow Lite Model Maker and Android Studio ML Model Binding

```
!pip install -q tflite-model-maker
```

```
     |████████████████████████████████| 577 kB 3.7 MB/s 
     |████████████████████████████████| 3.4 MB 42.3 MB/s 
     |████████████████████████████████| 1.1 MB 47.1 MB/s 
     |████████████████████████████████| 238 kB 52.1 MB/s 
     |████████████████████████████████| 10.9 MB 18.3 MB/s 
     |████████████████████████████████| 60.9 MB 121 kB/s 
     |████████████████████████████████| 840 kB 31.4 MB/s 
     |████████████████████████████████| 1.3 MB 6.0 MB/s 
     |████████████████████████████████| 87 kB 1.3 MB/s 
     |████████████████████████████████| 128 kB 70.7 MB/s 
     |████████████████████████████████| 77 kB 2.7 MB/s 
     |████████████████████████████████| 25.3 MB 47.0 MB/s 
     |████████████████████████████████| 497.9 MB 32 kB/s 
     |████████████████████████████████| 352 kB 76.9 MB/s 
     |████████████████████████████████| 1.4 MB 55.3 MB/s 
     |████████████████████████████████| 5.8 MB 28.5 MB/s 
     |████████████████████████████████| 462 kB 51.6 MB/s 
     |████████████████████████████████| 40 kB 5.8 MB/s 
     |████████████████████████████████| 1.1 MB 56.0 MB/s 
     |████████████████████████████████| 216 kB 60.0 MB/s 
  Building wheel for fire (setup.py) ... done
```

```python
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')

import matplotlib.pyplot as plt
import numpy as np
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

````
Mounted at /content/drive
````

```python
import os
 
# specify your path of directory
path = r"/content/drive/MyDrive/bbb"
 
# call listdir() method
# path is a directory of which you want to list
directories = os.listdir( path )
 
# This would print all the files and directories
for file in directories:
   print(file)
```

```
Fig 
Apple 
Watermelon 
Lemon 
Orange 
GrapeBlue
```

```py
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/content/drive/MyDrive/bbb/Apple/100_100.jpg')
plt.imshow(img)
```

```
<matplotlib.image.AxesImage at 0x7f0e3b934b90>
```

![image-20250425090118844](http://pdm888.oss-cn-beijing.aliyuncs.com/img/image-20250425090118844.png) 

```python
import os
from os import listdir
 
# get the path/directory
folder_dir = "/content/drive/MyDrive/bbb"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        print(images)

```

```python
path1 = os.path.abspath('/content/drive/MyDrive/bbb/Fig')
path2 = os.path.abspath('/content/drive/MyDrive/bbb/Apple')
path3 = os.path.abspath('/content/drive/MyDrive/bbb/Watermelon')
path4 = os.path.abspath('/content/drive/MyDrive/bbb/Lemon')
path5 = os.path.abspath('/content/drive/MyDrive/bbb/Orange')
path6 = os.path.abspath('/content/drive/MyDrive/bbb/GrapeBlue')
folder = os.path.join(path1, path2, path3, path4, path5, path6)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
            return images

print(load_images_from_folder(folder))
```

```
[array([[[253, 255, 255],
        [253, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],

       [[253, 255, 255],
        [253, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],

       ...,
...
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)]
        Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

```python
data = DataLoader.from_folder("/content/drive/MyDrive/bbb")
train_data, test_data = data.split(0.9)
```

```python
Fruitmodel = image_classifier.create(train_data)
```

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
 rasLayerV1V2)                                                   
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 6)                 7686      
                                                                 
=================================================================
Total params: 3,420,710
Trainable params: 7,686
Non-trainable params: 3,413,024
_________________________________________________________________
None
Epoch 1/5
33/33 [==============================] - 301s 9s/step - loss: 0.8611 - accuracy: 0.8030
Epoch 2/5
33/33 [==============================] - 39s 1s/step - loss: 0.4823 - accuracy: 1.0000
Epoch 3/5
33/33 [==============================] - 39s 1s/step - loss: 0.4536 - accuracy: 1.0000
Epoch 4/5
33/33 [==============================] - 39s 1s/step - loss: 0.4466 - accuracy: 1.0000
Epoch 5/5
33/33 [==============================] - 39s 1s/step - loss: 0.4442 - accuracy: 1.0000
```

```
loss, accuracy = Fruitmodel.evaluate(test_data)
```

```
4/4 [==============================] - 36s 8s/step - loss: 0.4295 - accuracy: 1.0000
```

```
Fruitmodel.export(export_dir='.')
```

```
/usr/local/lib/python3.7/dist-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "
```

```
from google.colab import files
files.download('model.tflite') 
```



---

---

# Test Module

![image-20250425090647518](http://pdm888.oss-cn-beijing.aliyuncs.com/img/image-20250425090647518.png) 



![Screenshot_20221108-194955](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-194955.png)

![Screenshot_20221108-195209](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-195209.png)

![Screenshot_20221108-195234](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-195234.png)

![Screenshot_20221108-195251](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-195251.png)

![Screenshot_20221108-195305](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-195305.png)

![Screenshot_20221108-195318](http://pdm888.oss-cn-beijing.aliyuncs.com/img/Screenshot_20221108-195318.png)

