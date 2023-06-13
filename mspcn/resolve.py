from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time
import scipy.io as sio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start = time.time()

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model', type=str, required=False, help='model file to use', default='mspcn/model_epoch_60.pth')
parser.add_argument('--cuda', action='store_true', help='use cuda', default=True)
opt = parser.parse_args()

print(opt)

model = torch.load(opt.model)

if opt.cuda:
    model = model.cuda()

root_path = "./mspcn/"
files = [f for f in os.listdir(root_path) if f.__contains__('.mat')]
images = sio.loadmat(root_path+str(files[0]))['normal_rat_kidney']
images = images.swapaxes(1,-1).swapaxes(0,1)

for img in images:  
    img = img.reshape([img.shape[1],img.shape[0],1])

    input = Variable(torch.tensor(img).float()).view(1, -1, img.shape[1], img.shape[0])

    if opt.cuda:
        input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img = out.data[0].numpy()
    out_img = out_img.reshape(img.shape[1] * 4, img.shape[0] * 4)
    sio.savemat('normal_rat_kidney_re/'+str(file), {'reconstruction':out_img})
    print('output image saved to ','normal_rat_kidney_re/'+str(file))
  

end = time.time()

print (end-start)

