from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mspcn.model import Net
from mspcn.dataset import get_training_set, get_test_set
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def matlab_style_gauss2D(shape=(7,7),sigma=1):
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h.astype('float32')
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    h = h.astype('float32')
    return h


if __name__ == '__main__':

    start = time.time()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')

    ##########   DATASET   ###########
    transform = transforms.Compose([
                    transforms.ToTensor()
                ])
    target_transform = transforms.Compose([
                    transforms.ToTensor()
                ])

    ###########   DATASET   ###########
    train_set = get_training_set(transform=None, target_transform=None)
    test_set = get_test_set(transform=None, target_transform=None)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


    print('===> Building model')
    model = Net(upscale_factor=opt.upscale_factor)
    print(model)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion2 = criterion2.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)


    psf_heatmap = torch.from_numpy(matlab_style_gauss2D(shape=(7,7),sigma=1))
    gfilter = torch.reshape(psf_heatmap, [1, 1, 7, 7])
    zero = torch.zeros(opt.batchSize, 1, 128, 128)
    amplitude = 50.0
    lmd = 1.0
    if cuda:
        gfilter = gfilter.cuda()
        zero = zero.cuda()


    def train(epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target = Variable(batch[0].float()), Variable(batch[1].float())
            if cuda:
                input = input.cuda()
                target = target.cuda()

            target *= amplitude
            target = F.conv2d(target, gfilter)
            optimizer.zero_grad()
            out = model(input)
            out2 = F.conv2d(out, gfilter)
            loss = criterion(out2, target) + lmd * criterion2(out, zero)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

        scheduler.step()
        
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


    def test():
        avg_psnr = 0
        for batch in testing_data_loader:
            input, target = Variable(batch[0].float()), Variable(batch[1].float())
            if cuda:
                input = input.cuda()
                target = target.cuda()

            target *= amplitude
            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(amplitude / mse.item())
            avg_psnr += psnr
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


    def checkpoint(epoch):
        model_out_path = "model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    if  __name__ == "__main__":
        for epoch in range(1, opt.nEpochs + 1):
            train(epoch)
            test()
            checkpoint(epoch)

    end = time.time()

    print (end-start)
