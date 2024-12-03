import glob
import tqdm
from natsort import natsort
import argparse
import options.options as option
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import os
import cv2
import time
def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def load_model(conf_path, model_path):
    opt = option.parse(conf_path)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    # model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt

def t_tensor(array): 
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(img.shape)==3:
        cv2.imwrite(path, img[:, :, [2, 1, 0]])
    else:
        cv2.imwrite(path, img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="./confs/TaterFlow_3b6f.yml")
    parser.add_argument("-n", "--name", default="unpaired")
    parser.add_argument("-model_path", default="../experiments/checkpoint.pth")
    # parser.add_argument("-data_root", default="../example")
    parser.add_argument("-data_root", default="../example_try")
    parser.add_argument("-resize_back", default=True)
    parser.add_argument("-resize", default=512)
    args = parser.parse_args()
    model, opt = load_model(args.opt,args.model_path)
    lr_paths = fiFindByWildcard(os.path.join(args.data_root,'input', '*.*'))
    depth_paths = fiFindByWildcard(os.path.join(args.data_root,'depth', '*.*'))
    grad_paths = fiFindByWildcard(os.path.join(args.data_root,'grad', '*.*'))
    t_paths = fiFindByWildcard(os.path.join(args.data_root,'T_pro', '*.*'))
    test_rgb_dir = os.path.join(args.data_root, 'results')
    if not os.path.exists(test_rgb_dir):
        os.makedirs(test_rgb_dir)
    times = []
    for lr_path, depth_path, grad_path, t_path, idx_test in tqdm.tqdm(zip(lr_paths,depth_paths,grad_paths,t_paths, range(len(lr_paths)))):
        lr = imread(lr_path)
        h,w,c = lr.shape
        depth = cv2.cvtColor(imread(depth_path),cv2.COLOR_BGR2GRAY)#[...,np.newaxis]
        grad = cv2.cvtColor(imread(grad_path),cv2.COLOR_BGR2GRAY)#[...,np.newaxis]
        depth = cv2.resize(depth, (args.resize, args.resize))
        grad = cv2.resize(grad, (args.resize, args.resize))
        lr = cv2.resize(lr, (args.resize, args.resize))
        grad = grad[...,np.newaxis]
        depth = depth[...,np.newaxis]
        lr = t_tensor(lr)
        depth = t_tensor(depth)
        grad = t_tensor(grad)   
        inputs = torch.cat((lr, lr, grad, depth), 1).to(opt['gpu_ids'][0])
        with torch.cuda.amp.autocast():
            sr = model.get_sr(lq = inputs, heat=None)
        sr = rgb(torch.clamp(sr, 0, 1))
        if args.resize_back:
            sr = cv2.resize(sr,(w,h))

        path_out_enhance = os.path.join(test_rgb_dir, os.path.basename(lr_path))
        imwrite(path_out_enhance, sr)


if __name__ == "__main__":
    main()
