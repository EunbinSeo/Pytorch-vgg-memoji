import torch
import torch.nn
import cv2
import torchvision.models as models

import util
import files
import compare


refsFolder = "pics/obama"
refsPattern = ".png"

grabRegion="145,70,190,190"
grabFileName="screen.png"

outFolder="out"

torch.manual_seed(42)

net = models.load_state_dict(torch.load('./torch_model/VGG_FACE.t7'))
net.eval()

