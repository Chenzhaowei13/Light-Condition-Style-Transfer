import os
from erf_settings import *
import numpy as np
from tools import prob_to_lines as ptl
import cv2
import models
import torch
import torch.nn.functional as F
from options.options import parser
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import numpy

cap_name = '/home/chen/data/video/test.mp4'
image = './data/00000.jpg'

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)
    # model
    model = models.ERFNet(5)
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    if args.mode == 0:  # mode 0 for video
        cap = cv2.VideoCapture(cap_name)
        while(True):
            check, in_frame_src = cap.read()
            if check:
                test(model, in_frame_src)
            else:
                print("Last frame")
                break

    elif args.mode == 1: # mode 1 for test image
        image_src = cv2.imread(image)
        test(model, image_src)
        cv2.waitKey(0)

def test(model, image_src):

    in_frame_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    # Input
    in_frame = cv2.resize(in_frame_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    croppedImage = in_frame[VERTICAL_CROP_SIZE:, :, :]  # FIX IT
    croppedImageTrain = cv2.resize(croppedImage, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)

    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    )

    image = input_transform(croppedImageTrain)
    image = image.unsqueeze(0)
    input_var = torch.autograd.Variable(image)

    # Comput
    output, output_exist = model(input_var)
    output = F.softmax(output, dim=1)
    pred = output.data.cpu().numpy()  # BxCxHxW
    pred_exist = output_exist.data.cpu().numpy()

    maps = []
    mapsResized = []
    exists = []
    img = Image.fromarray(cv2.cvtColor(croppedImageTrain, cv2.COLOR_BGR2RGB))

    for l in range(LANES_COUNT):
        prob_map = (pred[0][l + 1] * 255).astype(int)
        prob_map = cv2.blur(prob_map, (9, 9))
        prob_map = prob_map.astype(np.uint8)
        maps.append(prob_map)
        mapsResized.append(cv2.resize(prob_map, (IN_IMAGE_W, IN_IMAGE_H_AFTER_CROP), interpolation=cv2.INTER_LINEAR))
        img = ptl.AddMask(img, prob_map, COLORS[l])  # Image with probability map

        exists.append(pred_exist[0][l] > 0.5)
        lines = ptl.GetLines(exists, maps)

    print(exists)
    res_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("result_pb", res_img)

    for l in range(LANES_COUNT):
        points = lines[l]  # Points for the lane
        for point in points:
            cv2.circle(image_src, point, 5, POINT_COLORS[l], -1)

    cv2.imshow("result_points", image_src)
    cv2.waitKey(100)


if __name__ == '__main__':
    main()