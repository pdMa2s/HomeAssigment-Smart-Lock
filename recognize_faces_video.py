# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2
from utils.media_sources import VideoFileSource, VideoStreamSource
from imutils.video import FPS

detection_method_choices = ["hog", "cnn", "haar"]


def parse_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    stream_group = ap.add_mutually_exclusive_group()
    stream_group.add_argument("-v", "--video-stream", action='store_true',
                              help="Use the available video stream instead of video file")
    stream_group.add_argument("-vf", "--video-file", type=str, help="path to inout video",
                              default="test_videos/welcome_scene.mp4")

    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")

    ap.add_argument("-o", "--output", type=str,
                    help="path to output video", default="")
    ap.add_argument("-y", "--display", action='store_true', help="whether or not to display output frame to screen")
    ap.add_argument("-d", "--detection-method", type=str, default="hog", choices=detection_method_choices,
                    help="face detection model to use: either `hog` or `cnn`")
    return vars(ap.parse_args())


args = parse_arguments()

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# writer = None

if args["video_stream"]:
    video_source = VideoStreamSource()
else:
    video_source = VideoFileSource(args["video_file"])


i = 0
try:
    fps = FPS().start()
    while video_source.is_open():
        frame = video_source.get_frame()

        # This condition prevents from infinte looping
        # incase video ends.
        if video_source.is_stream_over():
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            voted_name = "Unknown"
            # check to see if we have found a match
            counts = {}
            for i, _ in filter(lambda b: b[1], enumerate(matches)):
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                voted_name = max(counts, key=counts.get)

            # update the list of names
            names.append(voted_name)

        if args["display"]:
            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        fps.update()

    fps.stop()

finally:
    # do a bit of cleanup
    cv2.destroyAllWindows()
    video_source.close()


print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))