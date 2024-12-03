from classifier import classifyVideo
from detect import localise
from makevideo import stitch
from utils import annotateVideo
import os

def main():
    #Set paths
    videoPath = 'trimmedFinal.mp4'
    cropDir = 'trackedCrops'
    stitchedDir = 'stitchedVideos'
    annotatedVideo = 'annotatedVideo.mp4'

    #Step 1: Localise actors in the video
    print('Running YOLO object detector and Byte Tracker...')
    if not os.path.exists(videoPath):
        raise FileNotFoundError(f'Video file not found: {videoPath}')
    bboxes = localise(videoPath, cropDir)
    print('Done.')

    #Step 2: Stitch cropped frames into videos for each actor
    print('Stitching cropped frames into videos...')
    stitch(cropDir, stitchedDir, hasSubdir=True)
    print('Done.')

    #Step 3: Classify each actor's video
    print('Classifying actor videos using SlowFast...')
    predictions = {}
    for stitchedVideo in os.listdir(stitchedDir):
        if stitchedVideo.endswith('.mp4'):
            path = os.path.join(stitchedDir, stitchedVideo)
            try:
                actorID = int(stitchedVideo.split('_'[1].split('.')[0]))
                print(f'Actor {actorID} video: {path}')
            except (IndexError, ValueError):
                print(f'Skipping {stitchedVideo} - Invalid actor ID')
                continue

            try:
                predLabels = classifyVideo(path)
                if not predLabels:
                    print(f'No predictions for {path}, assigning Unknown')
                    predLabels = ["Unknown"]
                predictions[actorID] = predLabels[0]
                print(f'Actor {actorID}: {predLabels[0]}')
            except Exception as e:
                print(f'Classification failed for {path}: {e}')
                predictions[actorID] = "Unknown"
    print('Done classifying.')

    #Step 4: Annotate the original video with predictions
    print('Annotating original video...')
    annotateVideo(videoPath, annotatedVideo, predictions, bboxes=bboxes)
    print(f'Annotated video saved at: {annotatedVideo}')
    print('Done.')

if __name__ == "__main__":
    main()