import time
from imutils.video import VideoStream
from abc import abstractmethod, ABC
import cv2


class MediaSource(ABC):

    @abstractmethod
    def is_open(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def is_stream_over(self) -> bool:
        pass


class VideoFileSource(MediaSource):
    def __init__(self, file_path: str):
        self.cap = cv2.VideoCapture(file_path)
        self.stream_over = False

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def close(self):
        self.cap.release()

    def get_frame(self):
        self.stream_over, frame = self.cap.read()
        return frame

    def is_stream_over(self):
        return not self.stream_over


class VideoStreamSource(MediaSource):
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        time.sleep(2)

    def is_open(self) -> bool:
        return True

    def close(self):
        self.vs.stop()

    def get_frame(self):
        return self.vs.read()

    def is_stream_over(self) -> bool:
        return False