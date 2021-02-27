"""Model predict."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 02月 27日 星期六 17:25:55 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from data import get_transform
from model import get_model, model_device

def normal_predict(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma - mi + 1e-12)

    return dn

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/ImageMatting.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/*.png", help="input image")
    parser.add_argument('--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = get_model(args.checkpoint)
    device = model_device()
    model.eval()

    totensor = get_transform(train=False)
    toimage = transforms.ToPILImage()

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total = len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_image = image.resize((320, 320))
        input_tensor = totensor(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor[:,0,:,:]
        output_tensor = normal_predict(output_tensor)

        output_image = toimage(output_tensor.cpu())
        output_image = output_image.resize((image.width, image.height))

        output_image.save("{}/{}".format(args.output, os.path.basename(filename)))
