from train_algorithm import train
from unet import UNet
from vgg import Vgg16
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("-e", "--epochs", type=int)
args = parser.parse_args()

classifier = Vgg16().cuda()
segmentor = UNet(7, 1000).cuda()

if args.epochs is not None:
    epochs = args.epochs
else:
    epochs = 10

train(classifier, segmentor, args.dir, epochs)