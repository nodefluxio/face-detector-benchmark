import numpy as np
import cv2
import dlib

class OpenCVHaarFaceDetector():
	def __init__(self,
		scaleFactor=1.3,
		minNeighbors=5,
		model_path='models/haarcascade_frontalface_default.xml'):

		self.face_cascade=cv2.CascadeClassifier(model_path)
		self.scaleFactor=scaleFactor
		self.minNeighbors=minNeighbors

	def detect_face(self,
		image):
		gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor,self.minNeighbors)

		faces=[[x,y,x+w,y+h] for x,y,w,h in faces]

		return np.array(faces)
		


class DlibHOGFaceDetector():
	def __init__(self,
		nrof_upsample=0,
		det_threshold=0):
		self.hog_detector=dlib.get_frontal_face_detector()
		self.nrof_upsample=nrof_upsample
		self.det_threshold=det_threshold

	def detect_face(self,
		image):

		dets, scores, idx = self.hog_detector.run(image, self.nrof_upsample, self.det_threshold)

		faces=[]
		for i, d in enumerate(dets):
			x1=int(d.left())
			y1=int(d.top())
			x2=int(d.right())
			y2=int(d.bottom())

			faces.append(np.array([x1,y1,x2,y2]))

		return np.array(faces)

class DlibCNNFaceDetector():
	def __init__(self,
		nrof_upsample=0,
		model_path='models/mmod_human_face_detector.dat'):

		self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
		self.nrof_upsample=nrof_upsample

	def detect_face(self,
		image):

		dets=self.cnn_detector(image,self.nrof_upsample)

		faces=[]
		for i, d in enumerate(dets):
			x1=int(d.rect.left())
			y1=int(d.rect.top())
			x2=int(d.rect.right())
			y2=int(d.rect.bottom())
			score=float(d.confidence)

			faces.append(np.array([x1,y1,x2,y2]))

		return np.array(faces)