# Benchmark of Publicly Available Face Model on WIDER Dataset

In this repository, we provide several face detection modules and models which you can use for your own application.
We also provided script to benchmark performance of each techniques on WIDER Face dataset

The available face detection techniques in this repositories are:

1. [OpenCV Haar Cascades Classifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
2. [DLib HOG](http://dlib.net/face_detector.py.html)
3. [DLib CNN](http://dlib.net/cnn_face_detector.py.html)
4. [Multi-task Cascaded CNN (Tensorflow)](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
5. [Mobilenet-SSD Face Detector (Tensorflow)](https://github.com/yeephycho/tensorflow-face-detection) 

## Dependencies
( This dependencies listed based on our testing machine, it is possible these scripts will work on the higher version of these dependencies )

* Python2.7
* OpenCV == 3.2.0
* Tensorflow == 1.4.1
* DLib == 19.10
* tqdm == 4.19.8

## WIDER Face dataset

You can download WIDER Face Dataset in [this link](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).
For benchmarking purpose, we only use train and validation split of the dataset because only those split which have
face bounding boxes ground truth information.
In order to use the benchmarking script, you should put the data under folder *dataset* like this structure :

```
+-- dataset
|   +-- WIDER_train
|   +-- WIDER_val
|   +-- wider_face_split
```

We perform several filtering on the dataset to avoid invalid image,(see [readme.txt](dataset/wider_face_split/readme.txt) under folder *dataset/wider_face_split*). Based on our assumption that the detected face is not informative if it is too small, we also discard ground truth bounding boxes with height and width less than 15 pixel.

## Running Benchmark Script

**DISCLAIMER** : In this repository, we didn't take account from which dataset the face detection model is trained (several method obviously performed better because they are trained using WIDER dataset). Our purpose in this repo is solely to show the reader which model is ready to be used and perform better on WIDER dataset. Maybe in the future we will also provide benchmark to the other dataset.

To perform performance benchmark on the available methods, you can execute the script below :
```
python wider_benchmark.py --method [selected-method] --iou_threshold [float between 0-1]
```
for the *--method* input you can select string input from the available method below:
1. opencv_haar
2. dlib_hog
3. dlib_cnn
4. mtcnn
5. mobilenet_ssd

The *--iou_threshold* input should be float value between 0 and 1. This iou threshold is used to determine whether the prediction box is correctly predicted that there is an object in that location. The prediction boxes will be assigned 'true prediction' if its *intersection over union (IoU)* with the ground truth boxes is bigger than *--iou_threshold* value. To read more about IoU, you can read it in [here](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/).

In term of performance, we benchmark the *mean average precision (mAP)* and *average IoU* metrices for each of the face detector methods. Using this script, you will obtain both metrices plus the average inferencing time per image in the dataset.

**IMPORTANT NOTICE** : The total amount of time spent for this script will be dependent on your machine. In our case, we spend around 30-45 minutes per method evaluated, by using appropriate hardware (for deep learning method (*dlib_cnn*,*mtcnn*,*mobilenet_ssd*), we are using GPU. Of course you can also try using CPU, but please don't be surprised when the estimated time processing all of the dataset is over **9 hours**, except the *mobilenet_ssd* which is fast enough on CPU)

## Our Benchmark Result

You can see our benchmark result in [here](benchmark-result.txt). For resource usage metrices, we are using python *memory profiler*

## Try on Your Own Project

You can use the available face detection module by inspecting the script below. Try running it with the available example image:

```
python test_deploy.py --input images/selfie.jpg --method mobilenet_ssd (or choose your desirable method)
```

Happy experimenting!!!

References:
1. https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
2. http://dlib.net/face_detector.py.html
3. http://dlib.net/cnn_face_detector.py.html
4. https://github.com/kpzhang93/MTCNN_face_detection_alignment
5. https://github.com/yeephycho/tensorflow-face-detection