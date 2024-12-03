import cv2

def annotateVideo(path, outputPath, predictions, bboxes, fps=None):
    """
    Overlay predictions on the original video for better understanding of the model output.
    Args:
        path (str): Path toe the original video.
        outputPath (str): Path to the output video.
        predictions (dict): A dictionary mapping actor IDs to their predicted labels.
            Example: {1: "Running", 2: "Walking", 3: "Standing"}
        bboxes (dict): A dictionary of bounding boxes for each frame and actor
            Example: {0: {1: (x1, y1, x2, y2), 2: (x1, y1, x2, y2), ...}, 1: {...}, ...}
        fps (int): Frames per second for the output video (default: None).
    """
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    originalFPS = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps else originalFPS

    print(f'Original video dimensions: {width}x{height} @ {originalFPS} FPS')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {frameCount}')

    #Videowriter setup
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(outputPath, FOURCC, fps, (width, height))
    frameIdx = 0
    writtenFrames = 0 #counts successfully written frames.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('End of video.')
            break
        #Overlay predictions on frame
        if frameIdx in bboxes:
            for actorID, bbox in bboxes[frameIdx].items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = predictions.get(actorID, "Unknown")
                cv2.putText(
                    frame,
                    f'Actor {actorID}: {label}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        output.write(frame)
        writtenFrames += 1
        frameIdx += 1
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    print(f'Annotated video saved to {outputPath}')
