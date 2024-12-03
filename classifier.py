import torch
import json
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
modenName = 'slowfast_r50'
model = torch.hub.load('facebookresearch/pytorchvideo', modenName, pretrained=True)
model = model.to(device)

with open("kinetics_classnames.json") as f:
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {int(v): k for k, v in kinetics_classnames.items()}
sideSize = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
numFrames = 32
cropSize = 256
samplingRate = 2
frames_per_ssecond = 30
alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()
    def forward(self, frames: torch.Tensor):
        fastPth = frames
        slowPth = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long()
        )
        return [fastPth, slowPth]
    
#Apply transformations to the video
transform = ApplyTransformToKey(
    key = 'video',
    transform = Compose(
        [
            UniformTemporalSubsample(samplingRate),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size = sideSize),
            CenterCropVideo(cropSize),
            PackPathway()
        ])
)

def classifyVideo(path: str):
    video = EncodedVideo.from_path(path)
    clipDuration = (numFrames * samplingRate) / frames_per_ssecond
    startSec = 0
    endSec = startSec + clipDuration
    videoData = video.get_clip(start_sec = startSec, end_sec = endSec)

    videoData = transform(videoData)
    inputs = videoData['video']
    inputs = [i.to(device)[None, ...] for i in inputs]

    #Get predictions
    with torch.no_grad():
        predictions = model(inputs)
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    #Get top 5 predictions
    top5 = predictions.topk(k=5).indices[0]
    predNames = [kinetics_id_to_classname[int(i)] for i in top5]
    return predNames