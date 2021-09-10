from abc import ABC, abstractmethod
import face_recognition
from typing import List, Tuple
import cv2


class FaceDetector(ABC):
    pass

    @abstractmethod
    def locate_face(self, rgb_image) -> List[Tuple[float, float, float, float]]:
        pass


class FaceRecognitionDetector(FaceDetector):
    available_methods: List[str] = ["hog", "cnn"]

    def __init__(self, method=available_methods[0]):
        assert method in self.available_methods
        self.detection_method = method

    def locate_face(self, rgb_image) -> List[Tuple[float, float, float, float]]:
        return face_recognition.face_locations(rgb_image, model=self.detection_method)


class HaarFaceDetector(FaceDetector):
    def __init__(self, haar_cascade_file_path: str = "haarcascade_frontalface_default.xml"):
        self.detector = cv2.CascadeClassifier(haar_cascade_file_path)

    def locate_face(self, rgb_image) -> List[Tuple[float, float, float, float]]:
        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        return [(y, x + w, y + h, x) for (x, y, w, h) in rects]