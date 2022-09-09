import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import get_loader, test_dataset, CHAOS
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
from torch.utils.data import DataLoader

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def getOneHotSegmentation(batch):
    backgroundVal = 0

    # Chaos MRI (These values are to set label values as 0,1,2,3 and 4)
    label1 = 0.24705882
    label2 = 0.49411765
    label3 = 0.7411765
    label4 = 0.9882353
    
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3, batch == label4),
                             dim=1)
    
    return oneHotLabels.float()

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()



class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))
        DiceZ = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])
            DiceZ[i, 0] = self.inter(pred[i, 4], GT[i, 4])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])
            DiceZ[i, 1] = self.sum(pred[i, 4], GT[i, 4])

        return DiceN, DiceB , DiceW, DiceT, DiceZ





def train(train_loader, model, optimizer, epoch, best_loss, device):
    
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    
    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    mseLoss = nn.MSELoss()

    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        InPhase, OutPhase, gt = pack
#         images = Variable(images).cuda()
#         gts = Variable(gts).cuda()
        if opt.same_input==True:
            output1_InPhase, , =model(InPhase, InPhase)
            output1_OutPhase, , =model(OutPhase, OutPhase)        
        else:
            output1_InPhase, , =model(InPhase, OutPhase)
            output1_OutPhase, , =model(OutPhase, InPhase)     
        #loss function
            
        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
        
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)# gt

        segmentation_prediction_ones = predToSegmentation(pred_y)# prediction
        
        Dice_loss = computeDiceOneHot()
        #then caculate loss of them

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)

        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.  
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        meanloss = test(model, opt.test_path)
        if meanloss < best_loss:
            print('new best loss: ', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth'% epoch)
    return best_loss


def test(model, test_loader, path):

    model.eval()
    mean_loss = []
    
    for i, pack in enumerate(train_loader, start=1):
        if opt.same_input==True:
            output1_InPhase, , =model(InPhase, InPhase)
            output1_OutPhase, , =model(OutPhase, OutPhase)        
        else:
            output1_InPhase, , =model(InPhase, OutPhase)
            output1_OutPhase, , =model(OutPhase, InPhase)     
        #loss function
            
        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
        
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)# gt

        segmentation_prediction_ones = predToSegmentation(pred_y)# prediction
        
        Dice_loss = computeDiceOneHot()
        
        #other metrics
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    for s in ['val', 'test']:
        image_root = '{}/data_{}.npy'.format(path, s)
        gt_root = '{}/mask_{}.npy'.format(path, s)
        test_loader = test_dataset(image_root, gt_root)

        dice_bank = []
        iou_bank = []
        loss_bank = []
        acc_bank = []

        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.cuda()

            with torch.no_grad():
                _, _, res = model(image)
            loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

            res = res.sigmoid().data.cpu().numpy().squeeze()
            gt = 1*(gt>0.5)            
            res = 1*(res > 0.5)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)
            acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            acc_bank.append(acc)
            
        print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
            format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

        mean_loss.append(np.mean(loss_bank))

    return mean_loss[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=25, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--device', type=str, default="cuda:1", help='gpu name')
    parser.add_argument('--same_input', type=bool, default="True", help='whether to use same input for Tranfuse')

    opt = parser.parse_args()

    # ---- build models ----
    model = TransFuse_S(pretrained=True).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = '{}/data_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_train.npy'.format(opt.train_path)
    
    
    train_set=CHAOS(isTrain=True)
    train_loader = DataLoader(train_set,
                          batch_size=opt.batch_size,
                          num_workers=opt.num_workers,
                          shuffle=False)
    
    test_set=CHAOS(isTrain=False)
    test_loader = DataLoader(test_set,
                      batch_size=opt.batch_size,
                      num_workers=opt.num_workers,
                      shuffle=False)

#     train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
#     total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    
    device=opt.device

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss, device)
