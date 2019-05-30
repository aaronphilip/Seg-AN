from train_algorithm import train
from unet import UNet
from vgg import Vgg16
from torch.cuda import set_device
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-d", "--device", type=int)
args = parser.parse_args()

if args.device is not None:
    set_device(args.device)

classifier = Vgg16().cuda()
segmentor = UNet(7, 1000).cuda()

if args.epochs is not None:
    epochs = args.epochs
else:
    epochs = 10

train(classifier, segmentor, epochs)