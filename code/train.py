import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
from utils.util import *
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import time
import options.options as option
from utils import util
from data import create_dataloader
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths
from data.LoL_dataset import LoL_Dataset, LoL_Dataset_v2
from torchvision.utils import save_image
import torchvision.transforms as T
# from visdom import Visdom
# viz = Visdom(env='ZZX_underwaterAAA21-512')  #启用可视化工具
# viz.line([0.], [0], win = 'loss', opts = dict(title = 'loss'))
to_tensor = T.ToTensor()
to_cv2_image = lambda x: np.array(T.ToPILImage()(torch.clip(x, 0, 1)))
torch.cuda.set_device(0)

def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_deviceDistIterSampler(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def color_adjust(low_light, output, kernel_size=7):
    # low_light, output = to_tensor(low_light), to_tensor(output)
    mean_kernal = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    low_light_mean = mean_kernal(low_light)
    output_mean = mean_kernal(output)
    color_align_output = output * (low_light_mean / output_mean)
    return color_align_output  # to_cv2_image(color_align_output)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, help = 'Path to option YMAL file.',
                            default = './confs/LOL_smallNet.yml') #  './confs/LOLv2-pc_rebuttal.yml') # 
    parser.add_argument('--launcher', choices = ['none', 'pytorch'], default = 'none',
                        help='job launcher')
    parser.add_argument('--local_rank', type = int, default=0)
    parser.add_argument('--tfboard', action = 'store_true')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train = True)

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

 

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    # if rank <= 0:
    #     logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt['dataset'] == 'LoL':
        dataset_cls = LoL_Dataset
    elif opt['dataset'] == 'LoL_v2':
        dataset_cls = LoL_Dataset_v2
    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = dataset_cls(opt = dataset_opt, train = True, all_opt = opt)
            train_loader = create_dataloader(True, train_set, dataset_opt, opt, None)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        elif phase == 'val':
            val_set = dataset_cls(opt = dataset_opt, train=False, all_opt=opt)
            val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    #### create model
    current_step = 0 #if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)
    print("Parameters of full network %.4f and encoder %.4f"%(sum([m.numel() for m in model.netG.parameters()])/1e6, sum([m.numel() for m in model.netG.RRDB.parameters()])/1e6))
    #### resume training
    current_step = 0
    start_epoch = 0
    # if resume_state:
    #     logger.info('Resuming training from epoch: {}, iter: {}.'.format(
    #         resume_state['epoch'], resume_state['iter']))
    # start_epoch = resume_state['epoch']
    # current_step = resume_state['iter']
    # model.resume_training(resume_state)  # handle optimizers and schedulers
    #####################   load pretrained model ###################################################
    # load_net = torch.load('..\\experiments\\train_noduibi_1121\\models\\3000_G.pth',map_location='cpu')
    # model.netG.load_state_dict(load_net)
    # print("load_done")
    # load_net = torch.load('../experiments/train_231112/models/100000_G_ori.pth',map_location='cpu')
    # load_net = torch.load('../experiments/65600_21.936446843984278_G.pth',map_location='cpu')
    # load_net = torch.load('../experiments/no_style_37200_22.00991090722511_G.pth',map_location='cpu')
    # load_net = torch.load('../experiments/train_withSF/models/127200_21.816189398883726_G.pth',map_location='cpu')
    load_net = torch.load('../experiments/train_withSF/models/127200_21.816189398883726_G.pth',map_location='cpu')
    model_path = '../experiments/33600_22.0638_0.9048_G.pth'
    model.netG.load_state_dict(load_net)
    print("load_done")
    #################################################################################################
    #### training
    timer = Timer()
    # logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()
    avg_psnr = best_psnr = -1
    # total_epochs=total_epochs*3
    
    psnr_best=0
    for epoch in range(start_epoch, total_epochs + 1):
        psnr_val = []
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        num_nan=0
        with torch.no_grad():
            for _, val_data in enumerate(val_loader):
                # print(val_data['GT'][0])
                # if val_data['GT_path'][0]!='11052':
                #     continue
                model.feed_data(val_data)
                # print、(val_data['LQ_path'][0])
                # print(val_data['LQ_path'][0],str(val_data['GT'].shape))
                
                sr = model.test()
                gt = val_data['GT'].to(sr.device)
                # print(val_data['GT_path'])
                # print(val_data['GT_path'])
                # print(str(torch.max(sr).item()))
                if str(torch.max(sr).item())=='nan':
                    print(val_data['GT_path'])
                    num_nan += 1
                # left=val_data['left'][0]
                # right=val_data['right'][0]
                # top=val_data['top'][0]
                # bottom=val_data['bottom'][0]
                # gt = gt[:,:,top:gt.shape[-2]-bottom,left:gt.shape[-1]-right]
                # sr = sr[:,:,top:sr.shape[-2]-bottom,left:sr.shape[-1]-right]
                temp_psnr, temp_ssim, N = compute_psnr_ssim(sr, gt)
                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                psnr_val.extend(to_psnr(sr, gt))
                sr = cv2.cvtColor((sr[0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
                if not os.path.exists(os.path.join('../results/', str(current_step))):
                    os.makedirs(os.path.join('../results/', str(current_step)))
                cv2.imwrite(os.path.join('../results/', str(current_step), val_data['LQ_path'][0]+'.png'),sr)
            # psnr_now = np.mean(np.array(psnr_val))
            # ssim_now = np.mean(np.array(val_ssim))
            # print(str(val_psnr.avg),' ',str(val_ssim.avg),' num_nan: ',str(num_nan))
            # time.sleep(1000)
            if val_psnr.avg > psnr_best:
                model.save(str(current_step) + '_' + str(val_psnr.avg)[:7]+ '_' + str(val_ssim.avg)[:6])
                print('best psnr ' + str(val_psnr.avg)[:7] +' ssim ' + str(val_ssim.avg)[:6] +' in step ' + str(current_step))
                psnr_best = val_psnr.avg
            elif val_psnr.avg >21.9:
                print('psnr ' + str(val_psnr.avg)[:7] +' ssim ' + str(val_ssim.avg)[:6] +' in step ' + str(current_step))
                model.save(str(current_step) + '_' + str(val_psnr.avg)[:7]+ '_' + str(val_ssim.avg)[:6])
            elif val_ssim.avg>0.9:
                print('psnr ' + str(val_psnr.avg)[:7] +' ssim ' + str(val_ssim.avg)[:6] +' in step ' + str(current_step))
                model.save(str(current_step) + '_' + str(val_psnr.avg)[:7]+ '_' + str(val_ssim.avg)[:6])
            else:
                print('psnr ' + str(val_psnr.avg)[:7] +' ssim ' + str(val_ssim.avg)[:6] +' in step ' + str(current_step))
            
        # print("Epoch: "+str(epoch))
        l2_loss_f = []
        vgg_loss_f = []
        ssim_loss = []
        style_loss = []
        lcr_loss=[]
        loss_total = 0
        timerData.tick()
        
        
        
        
        for _, train_data in enumerate(train_loader):
            timerData.tock()
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            # print("aaaaaa")
            model.feed_data(train_data)
            #### update learning rate
            # print("bbbbbb")
            
            # print(train_data['LQ'].shape)
            if current_step < 2:
                nll = 0
                sr = 0
            else:
                nll, sr, output_lr,t0 = model.optimize_parameters(current_step)
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            loss_total += nll
            #### log
            def eta(t_iter):
                return (t_iter * (opt['train']['niter'] - current_step )) / 3600
            # if current_step % opt['logger']['print_freq'] == 0 \
            #         or current_step - (resume_state['iter'] if resume_state else 0) < 25:
            # if current_step % 5 == 0:
            #     print(current_step)
            # Reduce number of logs
            if current_step % 20 == 0 or current_step  == 2:
                # viz.images(sr[0]*255, nrow=4, win='HR', opts={'title': 'HR'})
                # viz.images(train_data['LQ'][0,0:3]*255, nrow=4, win='Input', opts={'title': 'Input'})
                # viz.images(train_data['LQ'][0,3:6]*255, nrow=4, win='Input_his', opts={'title': 'Input_his'})
                # viz.images(train_data['GT'][0]*255, nrow=4, win='gt', opts={'title': 'gt'})
                # viz.line([loss_total/20], [current_step/ len(train_loader)], win='loss', update='append')
                sr = cv2.cvtColor((sr[0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
                input= cv2.cvtColor((train_data['LQ'][0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
                depth= cv2.cvtColor((train_data['depth'][0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_GRAY2BGR)
                grad= cv2.cvtColor((train_data['grad'][0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_GRAY2BGR)
                gt_t= cv2.cvtColor((train_data['t'][0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_GRAY2BGR)
                t0= cv2.cvtColor((t0[0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_GRAY2BGR)
                output_lr= cv2.cvtColor((output_lr[0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
                gt= cv2.cvtColor((train_data['GT'][0]*255).permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
                up0 = np.concatenate((depth,grad),1)
                up1 = np.concatenate((input,gt),1)
                down1  = np.concatenate((output_lr,sr),1)
                down2  = np.concatenate((gt_t,t0),1)
                visual = np.concatenate((up0,up1,down1,down2),0)
                cv2.imwrite('output.jpg',visual)
                # print('Iter: ' + str(current_step / len(train_loader)) +' loss: ' + str(loss_total / 100))
                # loss_total = 0
                # tb_logger_train.add_scalar('loss/nll', nll, current_step)
                # tb_logger_train.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                # tb_logger_train.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                # tb_logger_train.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                # tb_logger_train.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                # for k, v in model.get_current_log().items():
                #     tb_logger_train.add_scalar(k, v, current_step)
            #### save models and training states
            # if current_step % 10000 == 0:
            #     if rank <= 0:
            #         # logger.info('Saving models and training states.')
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)
            ### save best model
            # if avg_psnr > best_psnr:
            #     # logger.info('Saving best models')
            #     model.save('best_psnr')
            #     best_psnr = avg_psnr
                # model.save_training_state(epoch, current_step)
            timerData.tick()
        l2_loss_f.append(model.l2_loss_f.item())
        vgg_loss_f.append(model.vgg_loss_f.item())
        ssim_loss.append(model.ssim_loss.item())
        style_loss.append(model.style_loss.item())
        lcr_loss.append(model.lcr_loss.item())
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, l2:{:.2e}, per:{:.2e}, ssim:{:.2e}, style:{:.2e}, con:{:.2e}, nll:{:.3e}> '.format(
            epoch, current_step, model.get_current_learning_rate(), np.mean(np.array(l2_loss_f)), 
                                np.mean(np.array(vgg_loss_f)), np.mean(np.array(ssim_loss)), np.mean(np.array(style_loss)), np.mean(np.array(lcr_loss)), nll)
        print(message)
        l2_loss_f = []
        vgg_loss_f = []
        ssim_loss = []
        style_loss = []
        lcr_loss= []
        # if epoch%3==0:
        #     model.save(str(current_step))
        



    
    



# with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
#     f.write("TRAIN_DONE")

# if rank <= 0:
#     logger.info('Saving the final model.')
#     model.save('latest')
#     logger.info('End of training.')


if __name__ == '__main__':
    main()
