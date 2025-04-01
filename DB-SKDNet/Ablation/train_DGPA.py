
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.SemiModel_drop import SemiModel_drop

import torch
from torchvision.transforms import ColorJitter, GaussianBlur
import kornia.augmentation as K


# def dp(img):
#     # 颜色抖动增强，将参数调小以实现弱增强
#     # color_jitter = ColorJitter(brightness=[0.1,0.5], contrast=[0.1,0.5], saturation=[0.1,0.5], hue=0.05)
#     color_jitter = ColorJitter(brightness=[0.1,0.5], contrast=[0.1,0.5], saturation=[0.1,0.5])
#     img = color_jitter(img)
    
# #     # 随机高斯模糊增强，减小核大小和sigma范围，实现弱增强
# #     gaussian_blur = GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
# #     img = gaussian_blur(img)

#     return img

import time
start=time.time()

def seed_everything(seed):  # 随机数种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 保存检查点
def save_checkpoint(epoch, model, ema_model, optimizer, best_iou, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth'))

# 加载检查点
def load_checkpoint(checkpoint_path, model, ema_model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    best_iou = checkpoint['best_iou']
    return start_epoch, best_iou

# 查找最新的检查点
def find_latest_checkpoint(save_path):
    checkpoint_files = [f for f in os.listdir(save_path) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        return None
    # 按 epoch 编号排序
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(save_path, latest_checkpoint)

def update_ema_variables(model, ema_model, alpha):  #alpha是啥意思
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

import glob

def clean_old_checkpoints(save_path, keep_last=5):
    """保留最近的 keep_last 个检查点，删除其他"""
    checkpoint_files = sorted(glob.glob(os.path.join(save_path, 'checkpoint_epoch_*.pth')),
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoint_files) > keep_last:
        for old_checkpoint in checkpoint_files[:-keep_last]:
            os.remove(old_checkpoint)
            print(f"Deleted old checkpoint: {old_checkpoint}")


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, student_logits, teacher_logits):
        soft_teacher = F.softmax(teacher_logits/self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits/self.temperature, dim=1)
        return self.kl_loss(soft_student, soft_teacher) * (self.temperature**2)
    

def train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2,
           data_name, save_path, net,ema_net, criterion, 
           semicriterion, optimizer, use_zszl, num_epoches):
    # # 初始化知识蒸馏损失
    kd_criterion = KnowledgeDistillationLoss()
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    net.train(True)
    ema_net.train(True)

    weak_aug = nn.Sequential(

        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),

        K.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0), p=0.3),

        K.RandomGaussianNoise(mean=0.0, std=0.03, p=0.3),

        ).cuda()

    strong_aug = nn.Sequential(

        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),

        K.RandomGaussianBlur(kernel_size=(5,5), sigma=(0.5, 2.0), p=0.5),

        K.RandomGaussianNoise(mean=0.0, std=0.1, p=0.5),

        K.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # 随机擦除，不影响位置

        ).cuda()
    
    epoch_loss = 0

    length = 0
    st = time.time()
    loss_semi=torch.zeros(1)#创建了一个包含单个元素的张量 loss_semi，并且将这个元素的值初始化为 0。
    
    with tqdm(total=len(train_loader), desc=f'Eps {epoch}/{num_epoches}', unit='img') as pbar:#使用了 tqdm 库来创建一个进度条，用于在训练过程中直观地显示当前训练轮次（epoch）的进度。
        for i, (A, B, mask,with_label) in enumerate(train_loader):
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            with_label=with_label.cuda()#这是一个布尔类型的张量，用于指示当前样本是否带有标签
            
            A_w = weak_aug(A)
            B_w = weak_aug(B)

            A_s = strong_aug(A)
            B_s = strong_aug(B)
            optimizer.zero_grad()
#-----------------------------1.非 zszl 半监督学习部分（仅对有标签样本进行训练）-------------------------------------------

            if use_zszl is False:

                if with_label.any():
                    preds = net(A[with_label], B[with_label])
                    loss = criterion(preds[0], Y[with_label]) + criterion(preds[1], Y[with_label])
                    Y=Y[with_label]
                else:

                    continue

#----------------------------2.zszl 半监督学习部分（损失计算分为两部分，一部分是有标签样本的损失，另一部分是无标签样本的半监督损失）------------------------------------------------
#----------------------------2.1.有标签样本损失计算------------------------------------------------
            else:#use_zszl 为 True
                if with_label.any():
                    preds = net(A,B)
                    loss = criterion(preds[0][with_label], Y[with_label])  + criterion(preds[1][with_label], Y[with_label])
                else:
                    loss=0
#--------------------------- 2.2.无标签样本损失计算-------------------------------------------------

            if use_zszl is True:

                #学生
                preds = net(A_w,B_w,use_dropout=False)
                preds_ms = net(A_s,B_s,use_dropout=False)

                preds_fw=net(A_w,B_w,use_dropout=True,dropout_strength="weak")
                preds_fs=net(A_w,B_w,use_dropout=True,dropout_strength="strong")

                # # preds_ms_fw=net(A_s,B_s,use_dropout=True,dropout_strength="weak")
                # preds_fs=net(A_s,B_s,use_dropout=True,dropout_strength="strong")


               #教师
                with torch.no_grad():#禁用梯度计算，因为教师模型（ema_net）的预测结果仅作为伪标签，不需要进行反向传播。
                    pseudo_attn,pseudo_preds = net(A[~with_label],B[~with_label])
                    pseudo_attn,pseudo_preds =  torch.sigmoid(pseudo_attn).detach(),torch.sigmoid(pseudo_preds).detach()

                #蒸馏损失
                # loss_semi = semicriterion(preds[0][~with_label], pseudo_attn) + semicriterion(preds[1][~with_label], pseudo_preds) #
                loss_semi_1 = semicriterion(preds[0][~with_label], pseudo_attn) + semicriterion(preds[1][~with_label], pseudo_preds) #
                loss_semi_2 = semicriterion(preds_ms[0][~with_label], pseudo_attn) + semicriterion(preds_ms[1][~with_label], pseudo_preds) #

                loss_semi_3 = semicriterion(preds_fw[0][~with_label], pseudo_attn) + semicriterion(preds_fw[1][~with_label], pseudo_preds) #
                loss_semi_4 = semicriterion(preds_fs[0][~with_label], pseudo_attn) + semicriterion(preds_fs[1][~with_label], pseudo_preds) #

                # #图像之间
                loss_semi_5 = kd_criterion(preds_ms[0][~with_label], preds[0][~with_label]) + kd_criterion(preds_ms[1][~with_label],preds[1][~with_label]) 
                # #解码器之间
                loss_semi_6 = kd_criterion(preds_fs[0][~with_label], preds_fw[0][~with_label]) + kd_criterion(preds_fs[1][~with_label], preds_fw[1][~with_label]) 

                # loss_semi_7 = semicriterion(preds_ms_fw[0][~with_label], preds_mw_fw[0][~with_label]) + semicriterion(preds_ms_fw[1][~with_label], preds_mw_fw[1][~with_label]) 
                # loss_semi_8 = semicriterion(preds_ms_fs[0][~with_label], preds_mw_fs[0][~with_label]) + semicriterion(preds_ms_fs[1][~with_label], preds_mw_fs[1][~with_label]) 

                # loss_semi =semicriterion(preds[1][~with_label],pseudo_preds) #test!!!!!!!!! loss_semi2
                loss=loss+0.2*(loss_semi_1+loss_semi_2+loss_semi_3+loss_semi_4)+0.1*(loss_semi_5+loss_semi_6)  #全监督损失+半监督损失，半监督系数默认为0.2，测试0.3，0.4，0.5！！
                

                mask = mask.to(with_label.device)

                Eva_train2.add_batch(mask[~with_label].cpu().numpy().astype(int),
                                     (preds[1][~with_label] > 0).cpu().numpy().astype(int))  # ~相反
            # ---- loss function ----

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha=0.99)  #0.9，0.995，0.999

            epoch_loss += loss.item()#将当前批次的损失累加到本轮总损失 epoch_loss 中。

            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            #对主模型的第二个预测输出 preds[1] 应用 Sigmoid 函数，并将结果二值化得到预测标签 pred。
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_train.add_batch(target, pred)#将真实标签 target 和预测标签 pred 添加到评估器 Eva_train 中进行评估。
            pbar.set_postfix(**{'LAll': loss.item(),'LSemi': loss_semi.item()}) #？
            pbar.update(1)
            length += 1
            #更新进度条的显示信息，包括总损失 LAll 和半监督损失 LSemi，并将进度条向前推进一个批次。

    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1], Eva_train2.Recall()[1], Eva_train2.F1()[1]))

    if use_zszl is True:
        print(
            'Epoch [%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epoches, \
                IoU, Pre, Recall, F1))
    print("Strat validing!")


    net.train(False)
    net.eval()
    ema_net.train(False)#将主模型 net 和 EMA 模型 ema_net 设置为评估模式
    ema_net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]

            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)

            preds_ema = ema_net(A, B)[1]
            Eva_val2.add_batch(target, (preds_ema>0).cpu().numpy().astype(int))
            length += 1
            #将真实标签和预测标签添加到评估器 Eva_val 和 Eva_val2 中进行评估。
            """
                这里到底是存net的参数还是ema_net的参数，都可以，看哪个精度高
            """
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))

    new_iou = IoU[1]    #存教师模型
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
     
        print('best_epoch', epoch)
        student_dir = save_path + '_train1_' + '_best_student_iou.pth'
        # 1. 先建立一个字典，保存三个参数：
        student_state = {'best_student_net ': net.state_dict(),
                 'optimizer ': optimizer.state_dict(),
                 ' epoch': epoch}
        # 2.调用torch.save():其中dir表示保存文件的绝对路径+保存文件名，如'/home/qinying/Desktop/modelpara.pth'
        torch.save(student_state, student_dir)
        torch.save(ema_net.state_dict(), save_path + '_train1_' + '_best_main_model.pth') #当student的精度最高的时候，同时存teacher的精度，然后用teacher的精度进行测试
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()



if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number') #修改这里！！！
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size') #修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=0.05, help='Proportion of the labeled images')#修改这里！！！
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')  #修改这里！！！
    parser.add_argument('--data_name', type=str, default='LEVIR', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='SemiModel_noema04',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default='./output/C2F-SemiCD/WHU-zszl4/')  # 半监督的模型保存路径！！
    # parser.add_argument('--save_path', type=str, default='./output/C2FNet/WHU/')  # 全监督的模型保存路径！！
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    
    opt = parser.parse_args()
    print('labeled ration=0.05,Ablation现在半监督损失函数系数为:0.2!') # 修改参数后不真实

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = '/home/xiongchuan/C2F-SemiCD-and-C2FNet-main/CDDataset/LEVIR-CD/train/'
        opt.val_root = '/home/xiongchuan/C2F-SemiCD-and-C2FNet-main/CDDataset/LEVIR-CD/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/home/xiongchuan/C2F-SemiCD-and-C2FNet-main/CDDataset/WHU-CD/train/'
        opt.val_root = '/home/xiongchuan/C2F-SemiCD-and-C2FNet-main/CDDataset/WHU-CD/val/'
        # opt.train_root = '/data/chengxi.han/data/WHU-CD-256-Semi/train/'
        # opt.val_root = '/data/chengxi.han/data/WHU-CD-256-Semi/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
        opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    elif opt.data_name == 'DSIFN':
        opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
        opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    elif opt.data_name == 'SYSU':
        opt.train_root = '/home/zhaobin/CDDataset/SYSU/train/'
        opt.val_root = '/home/zhaobin/CDDataset/SYSU/val/'
    elif opt.data_name == 'S2Looking':
        opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
        opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'
    elif opt.data_name == 'GoogleGZ':
        opt.train_root = '/data/chengxi.han/data/Google_GZ_CD256/train/'
        opt.val_root = '/data/chengxi.han/data/Google_GZ_CD256/val/'
    elif opt.data_name == 'LEVIRsup-WHUunsup':
        opt.train_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/train/'
        opt.val_root = '/data/chengxi.han/data/WHU-LEVIR-CD-256-Semi/val/'

    train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize,opt.train_ratio, num_workers=8, shuffle=True, pin_memory=False)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=6, shuffle=False, pin_memory=False)
    # train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize,opt.train_ratio, num_workers=0, shuffle=True, pin_memory=True)
    # val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=False, pin_memory=True)

    Eva_train = Evaluator(num_class = 2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    model=SemiModel_drop().cuda()
    ema_model=SemiModel_drop().cuda()


    # ema_model = SemiModel_drop().cuda()

    for param in ema_model.parameters():
        param.detach_()

    criterion = nn.BCEWithLogitsLoss().cuda() # loss
    semicriterion = nn.BCEWithLogitsLoss().cuda() # loss

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0
    start_epoch = 1

    # 自动恢复训练
    if opt.resume:
        latest_checkpoint = find_latest_checkpoint(save_path)
        if latest_checkpoint:
            print(f"Resuming from latest checkpoint: {latest_checkpoint}")
            start_epoch, best_iou = load_checkpoint(latest_checkpoint, model, ema_model, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    print("Start train...")


    for epoch in range(start_epoch, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # 可以先全用有标签的训练几个epoch，再进行半监督训练 !!!!
        if epoch < 5: #默认的为5，测试10，15，20
            use_zszl= False
            # print('labeled ration=0.05,Ablationresumer现在监督训练的次数为:20!')
        else:
            use_zszl= True
  
        # 全程ema=False，即一直只用有标签的进行训练，不进行半监督学习
        
        # use_zszl=False
        # ----------------------------------------------------------------------------------------------------
        Eva_train.reset()#重置混淆矩阵
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()
        train1(train_loader, val_loader, Eva_train,Eva_train2, Eva_val,Eva_val2, data_name, save_path, model,
              ema_model, criterion,semicriterion, optimizer,use_zszl, opt.epoch)  # 需要理解
        # -----------------------------------------------------------------------------------------------------
        lr_scheduler.step()
        
        save_checkpoint(epoch, model, ema_model, optimizer, best_iou, save_path)
        clean_old_checkpoints(save_path, keep_last=5)
        # print('现在的数据是：', args.data_name)


end=time.time()
print('程序训练train的时间为:',end-start)




























