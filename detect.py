import cv2
from ultralytics import YOLO
import os
import supervision as sv


def localise(videoPath, outputDir, cropSize=(224, 224)):
    #Load YOLO model
    model = YOLO("yolo11m.pt")
    #initialise byte tracker
    tracker = sv.ByteTrack()
    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    #Open video
    cap = cv2.VideoCapture(videoPath)
    frameIdx = 0
    bboxes = {} #A dictionary to store the bounding boxes of each actor
    cropWidth, cropHeight = cropSize
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #Detect objects
        frameH, frameW, _ = frame.shape
        results = model.predict(frame, conf=0.5, classes=[0], stream=False, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        #Update tracker with detections
        detections = tracker.update_with_detections(detections)

        #Initialise the bounding boxes for the current frame.
        bboxes[frameIdx] = {}
        for detection, trackerID in zip(detections, detections.tracker_id):
            bbox = detection[0]
            x1, y1, x2, y2 = map(int, bbox)
            #Store the BBs
            bboxes[frameIdx][trackerID] = (x1, y1, x2, y2)
            #Centroid method cropping
            centreX = (x1 + x2) // 2
            centreY = (y1 + y2) // 2
            startX = max(0, centreX - cropWidth // 2)
            startY = max(0, centreY - cropHeight // 2)
            endX = min(frameW, startX + cropWidth)
            endY = min(frameH, startY + cropHeight)

            crop = frame[startY:endY, startX:endX]
            cropResized = cv2.resize(crop, cropSize)
            #Create a seperate directory for each actor
            actorDir =  os.path.join(outputDir, f"actor_{trackerID}")
            if not os.path.exists(actorDir):
                os.makedirs(actorDir, exist_ok=True)
            filename = os.path.join(actorDir, f"frame_{frameIdx}.jpg")
            cv2.imwrite(filename, cropResized)
        frameIdx += 1
    cap.release()
    cv2.destroyAllWindows()
    return bboxes


if __name__ == "__main__":
    videoPath = "video.mp4"
    outputDir = "output"
    localise(videoPath, outputDir)
