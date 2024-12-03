import os
import cv2

def stitch(cropDir, outputDir, fps=None, hasSubdir=True):
    fps = fps if fps else 30
    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    if hasSubdir:
        print('Assuming subdirs exist inside the given directory path')
        for actorDir in os.listdir(cropDir):
            if not os.path.isdir(os.path.join(cropDir, actorDir)):
                raise ValueError(f"Expected a directory, got a file: {actorDir}")
            actorPath = os.path.join(cropDir, actorDir)
            if os.path.isdir(actorPath):
                outputPath = os.path.join(outputDir, f"{actorDir}.mp4")
                images = sorted([os.path.join(actorPath, img) for img in os.listdir(actorPath) if img.endswith('.jpg')])
                if not images:
                    print(f"No images found in {actorPath}")
                    continue
                frame = cv2.imread(images[0])
                height, width, layers = frame.shape
                FOURCC =  cv2.VideoWriter_fourcc(*'mp4v')
                output = cv2.VideoWriter(outputPath, FOURCC, fps, (width, height))
                for image in images:
                    frame = cv2.imread(image)
                    if frame is None:
                        print(f'Could not read image:  {image}')
                        continue
                    output.write(frame)
                output.release()
                print(f'Stitched {actorDir} into {outputPath}')
    else:
        images = sorted([os.path.join(cropDir, img) for img in os.listdir(cropDir) if img.endswith('.jpg')])
        if not images:
            print(f"No images found in {cropDir}")
            return
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape
        FOURCC =  cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(outputDir, FOURCC, fps, (width, height))
        for image in images:
            frame = cv2.imread(image)
            if frame is None:
                print(f'Could not read image:  {image}')
                continue
            output.write(frame)
            output.release()
            print(f'Stitched into {outputDir}')