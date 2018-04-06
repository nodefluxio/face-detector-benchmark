# Benchmark of Publicly Available Face Model on WIDER Dataset

In this repository, we provide several face detection modules and models which you can use for your own application.
We also provided script to benchmark performance of each techniques on WIDER dataset

The available face detection techniques in this repositories are:

1. [OpenCV Haar Cascades Classifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
2. [DLib HOG](http://dlib.net/face_detector.py.html)
3. [Dlib CNN](http://dlib.net/cnn_face_detector.py.html)
4. [Multi-task Cascaded CNN (Tensorflow)](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
5. [Mobilenet-SSD Face Detector (Tensorflow)](https://github.com/yeephycho/tensorflow-face-detection) 

## Dependencies
( This dependencies listed based on our testing machine, it is possible these scripts will work on the higher version of these dependencies )

* Python2.7
* OpenCV == 3.2.0
* Tensorflow == 1.4.1
* DLib == 19.10
* tqdm == 4.19.8