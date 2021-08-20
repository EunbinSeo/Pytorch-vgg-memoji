import torch
import torch.nn
import torchvision.models as models
import cv2
from functools import partial

from vggface import VggFace

import util
import files
import compare
import pickle

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


targetsFolder = "./pics/memoji"
targetsPattern = ".png"

refsFolder = "./pics/trump"
refsPattern = ".jpg"
net = VggFace()

net.load_state_dict(torch.load("./torch_model/vggface-9d491dd7c30312.pth"))

net.eval()

selectedLayer = 38
print("Comparing reference files: ", refsFolder, refsPattern, "with targets: ", targetsFolder)

refs = []
files_ = files.scandir(refsFolder, refsPattern)

for i, file_ in enumerate(files_):
    refImg = util.process(cv2.imread(refsFolder+"/"+file_))
    refImg = torch.from_numpy(refImg)
    refImg = refImg.unsqueeze(0)
    
    net.fc3.register_forward_hook(get_activation('fc3'))
    net.forward(refImg.float())
    output =  activation['fc3']
    refs.append(output)

results = []
files_ = files.scandir(targetsFolder, targetsPattern)
for i, file_ in enumerate(files_):
    results.append(compare.compareFile(selectedLayer, refs, targetsFolder, file_,net))

maxids = util.indexsortTable(results)
print(maxids)
print(files_)
for i in range(3):
    print("i: ", i, "maxval: ", results[maxids[i]], "file: ", files_[maxids[i]]) 
    img = cv2.imread(targetsFolder+"/"+files_[maxids[i]])
    legend = "result" + str(results[maxids[i]].item()) #"l: %i, #: %i, val: %0.3f", selectedLayer, i, results[maxids[i]])
    cv2.imshow(legend, img)
    k = cv2.waitKey(0)
    if k == 27: # esc key
        cv2.destroyAllWindow()
    else:
        pass

