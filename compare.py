import util
import cv2
import torch
import os

def compareTensors(refs, target, targetName):
    sum_ = 0
    if len(refs) == 0:
        print("no reference images")
        return
    
    for i in range(len(refs)):
        ref = refs[i]
        dotself = torch.matmul(ref , ref.T)
        sum_ = sum_ + torch.matmul(ref, target.T) / dotself

        '''
        Trying straight up distance.  Note: need to reverse sort max/min.
        sum = sum + torch.dist(ref,target)  
        Trying mean squred error.  Note: need to reverse sort max/min.
        local mse = nn.MSECriterion()
        mse.sizeAverage = false
        local loss = mse:forward(ref,target) print("loss=", loss)
        sum = sum + loss 
        note, max/min reversed
        '''
    return sum_ / len(refs)

def compareFile(selectedLayer, refs, targetsFolder, fileName, net):
    img = util.process(cv2.imread(targetsFolder+"/"+fileName))
    #net.forward(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    net.fc3.register_forward_hook(get_activation('fc3'))
    output = net(img.float())
    output =  activation['fc3']
    return compareTensors(refs, output, fileName)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
