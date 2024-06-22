import json
from torch import nn
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss
from toolbox.loss.FocalLoss import FocalLoss
from log import get_logger
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from toolbox.models.A__Paper.guass.BAMCCM_RRMCCM_DBF_S import Module
torch.backends.cudnn.benchmark = True

# DATASET = "Potsdam"
DATASET = "Vaihingen"
batch_size = 28
import argparse
parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="/media/user/shuju/XJ/configs/{}.json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()
with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    '''solving the batch==1'''
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True, drop_last=True)
criterion = nn.CrossEntropyLoss().cuda()
criterion_without = MscCrossEntropyLoss().cuda()
criterion_focal1 = FocalLossbyothers().cuda()
criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督
criterion_Focal = FocalLoss(class_num=6).cuda()
'''定义网络'''
net = Module().cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)

def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
best = [0.0]
size = (32, 32)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 100

model = 'BAMCCM_RRMCCM_DBF_S'
logdir = f'weight/{DATASET}/{model}({time.strftime("%Y-%m-%d-%H-%M")})'
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')
logger.info(f'Epochs:{epochs}  Batchsize:{batch_size}')

for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]

        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image, ndsm)
        loss = criterion_without(out[0], label) + criterion_focal1(out[0], label)

        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0
        optimizer.zero_grad()
        loss.backward()  # backpropagation to get gradient
        optimizer.step()  # update the weight
        train_loss = loss.item() + train_loss

    net = net.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            outVal = net(imageVal, ndsmVal)
            loss = criterion_without(outVal[0], labelVal) + criterion_focal1(outVal[0], labelVal)
            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            labelVal = labelVal.data.cpu().numpy()
            accval = accuary(outVal, labelVal)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch + 1:3d}/{epochs:3d} || trainloss:{train_loss / len(train_dataloader):.8f} valloss:{eval_loss / len(test_dataloader):.8f} || '
        f'Valid Acc:{acc / len(test_dataloader)} || spend_time:{time_str}')
    print(epoch_str + time_str)

    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))
    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        torch.save(net.state_dict(), './{}/{}-{}-loss.pth'.format(logdir, model, DATASET))
    # 每3轮保存一次权重
    if (epoch + 1) % 3 == 0:
        # best.append(acc / len(test_dataloader))
        # numloss = epoch
        # torch.save(net.state_dict(), './weight/GAGNet_WOMSAFB-{}-loss.pth'.format(DATASET))
        torch.save(net.state_dict(), './{}/{}:{}-{}-loss.pth'.format(logdir, epoch + 1, model, DATASET))

    logger.info(f'best Accuracy epoch:{numloss:3d}  || best Accuracy:{max(best)}')

    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)

    trainlosslistnumpy = np.array(trainlosslist_nju)
    vallosslistnumpy = np.array(vallosslist_nju)

    np.savetxt('trainloss_M2RESIZE_nlpr.txt', trainlosslistnumpy)
    np.savetxt('valloss_M2RESIZE_nlpr.txt', vallosslistnumpy)
