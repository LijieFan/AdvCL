from __future__ import print_function

import argparse
import numpy as np
import os, csv
from dataset import CIFAR10IndexPseudoLabelEnsemble
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

from utils import progress_bar, TwoCropTransformAdv
from losses import SupConLoss
import tensorboard_logger as tb_logger
from models.resnet_cifar_multibn_ensembleFC import resnet18 as ResNet18
import random
from fr_util import generate_high
from utils import adjust_learning_rate, warmup_learning_rate
import apex

# ================================================================== #
#                     Inputs and Pre-definition                      #
# ================================================================== #

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,  default='advcl_cifar10',
                    help='name of the run')
parser.add_argument('--cname', type=str,  default='imagenet_clPretrain',
                    help='')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size')
parser.add_argument('--epoch', type=int, default=1000,
                    help='total epochs')
parser.add_argument('--save-epoch', type=int, default=100,
                    help='save epochs')
parser.add_argument('--epsilon', type=float, default=8,
                    help='The upper bound change of L-inf norm on input pixels')
parser.add_argument('--iter', type=int, default=5,
                    help='The number of iterations for iterative attacks')
parser.add_argument('--radius', type=int, default=8,
                    help='radius of low freq images')
parser.add_argument('--ce_weight', type=float, default=0.2,
                    help='cross entp weight')

# contrastive related
parser.add_argument('-t', '--nce_t', default=0.5, type=float,
                    help='temperature')
parser.add_argument('--seed', default=0, type=float,
                    help='random seed')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--cosine', action='store_true',
                    help='using cosine annealing')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')
parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
args = parser.parse_args()
args.epochs = args.epoch
args.decay = args.weight_decay
args.cosine = True
import math
if args.batch_size > 256:
        args.warm = True
if args.warm:
    args.warmup_from = 0.01
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.name = 'AdvCL_Cifar10'
config = {
    'epsilon': args.epsilon / 255.,
    'num_steps': args.iter,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}
# ================================================================== #
#                      Data and Pre-processing                       #
# ================================================================== #
print('=====> Preparing data...')
# Multi-cuda
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
train_transform_org = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
transform_train = TwoCropTransformAdv(transform_train, train_transform_org)

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

label_pseudo_train_list = []
num_classes_list = [2, 10, 50, 100, 500]

dict_name = 'data/{}_pseudo_labels.pkl'.format(args.cname)
f = open(dict_name, 'rb')  # Pickle file is newly created where foo1.py is
feat_label_dict = pickle.load(f)  # dump data to f
f.close()
for i in range(5):
    class_num = num_classes_list[i]
    key_train = 'pseudo_train_{}'.format(class_num)
    label_pseudo_train = feat_label_dict[key_train]
    label_pseudo_train_list.append(label_pseudo_train)

train_dataset = CIFAR10IndexPseudoLabelEnsemble(root='data',
                                                transform=transform_train,
                                                pseudoLabel_002=label_pseudo_train_list[0],
                                                pseudoLabel_010=label_pseudo_train_list[1],
                                                pseudoLabel_050=label_pseudo_train_list[2],
                                                pseudoLabel_100=label_pseudo_train_list[3],
                                                pseudoLabel_500=label_pseudo_train_list[4],
                                                download=True)
# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=n_gpu*4)


# ================================================================== #
#                     Model, Loss and Optimizer                      #
# ================================================================== #

# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_t1, images_t2, images_org, targets, criterion):
        x1 = images_t1.clone().detach()
        x2 = images_t2.clone().detach()
        x_cl = images_org.clone().detach()
        x_ce = images_org.clone().detach()

        images_org_high = generate_high(x_cl.clone(), r=args.radius)
        x_HFC = images_org_high.clone().detach()

        if self.rand:
            x_cl = x_cl + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)
            x_ce = x_ce + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            x_cl.requires_grad_()
            x_ce.requires_grad_()
            with torch.enable_grad():
                f_proj, f_pred = self.model(x_cl, bn_name='pgd', contrast=True)
                fce_proj, fce_pred, logits_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
                f1_proj, f1_pred = self.model(x1, bn_name='normal', contrast=True)
                f2_proj, f2_pred = self.model(x2, bn_name='normal', contrast=True)
                f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
                features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
                loss_contrast = criterion(features)
                loss_ce = 0
                for label_idx in range(5):
                    tgt = targets[label_idx].long()
                    lgt = logits_ce[label_idx]
                    loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.
                loss = loss_contrast + loss_ce * args.ce_weight
            grad_x_cl, grad_x_ce = torch.autograd.grad(loss, [x_cl, x_ce])
            x_cl = x_cl.detach() + self.step_size * torch.sign(grad_x_cl.detach())
            x_cl = torch.min(torch.max(x_cl, images_org - self.epsilon), images_org + self.epsilon)
            x_cl = torch.clamp(x_cl, 0, 1)
            x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            x_ce = torch.clamp(x_ce, 0, 1)
        return x1, x2, x_cl, x_ce, x_HFC

print('=====> Building model...')
bn_names = ['normal', 'pgd', 'pgd_ce']
model = ResNet18(bn_names=bn_names)
model = model.cuda()
# tb_logger
if not os.path.exists('logger'):
    os.makedirs('logger')
logname = ('logger/pretrain_{}'.format(args.name))
logger = tb_logger.Logger(logdir=logname, flush_secs=2)
if torch.cuda.device_count() > 1:
    print("=====> Let's use", torch.cuda.device_count(), "GPUs!")
    model = apex.parallel.convert_syncbn_model(model)
    model = nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True
else:
    print('single gpu version is not supported, please use multiple GPUs!')
    raise NotImplementedError
net = AttackPGD(model, config)
# Loss and optimizer
ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
contrast_criterion = SupConLoss(temperature=args.nce_t)
optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay)
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# ================================================================== #
#                           Train and Test                           #
# ================================================================== #

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, targets, ind) in enumerate(train_loader):
        tt = []
        for tt_ in targets:
            tt.append(tt_.to(device).long())
        targets = tt
        image_t1, image_t2, image_org = inputs
        image_t1 = image_t1.cuda(non_blocking=True)
        image_t2 = image_t2.cuda(non_blocking=True)
        image_org = image_org.cuda(non_blocking=True)
        warmup_learning_rate(args, epoch+1, batch_idx, len(train_loader), optimizer)
        # attack contrast
        optimizer.zero_grad()
        x1, x2, x_cl, x_ce, x_HFC = net(image_t1, image_t2, image_org, targets, contrast_criterion)
        f_proj, f_pred = model(x_cl, bn_name='pgd', contrast=True)
        fce_proj, fce_pred, logits_ce = model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
        f1_proj, f1_pred = model(x1, bn_name='normal', contrast=True)
        f2_proj, f2_pred = model(x2, bn_name='normal', contrast=True)
        f_high_proj, f_high_pred = model(x_HFC, bn_name='normal', contrast=True)
        features = torch.cat(
                [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
        contrast_loss = contrast_criterion(features)
        ce_loss = 0
        for label_idx in range(5):
            tgt = targets[label_idx].long()
            lgt = logits_ce[label_idx]
            ce_loss += ce_criterion(lgt, tgt) / 5.

        loss = contrast_loss + ce_loss * args.ce_weight
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets[0].size(0)

        progress_bar(batch_idx, len(train_loader),
                     'Loss: %.3f (%d/%d)'
                     % (train_loss/(batch_idx+1), correct, total))

    return train_loss/batch_idx, 0.


# ================================================================== #
#                             Checkpoint                             #
# ================================================================== #

# Save checkpoint
def checkpoint(epoch):
    print('=====> Saving checkpoint...')
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    save_dir = './checkpoint/{}'.format(args.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, '{}/epoch_{}.ckpt'.format(save_dir, epoch))

# ================================================================== #
#                           Run the model                            #
# ================================================================== #

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

for epoch in range(start_epoch, args.epoch+2):
    adjust_learning_rate(args, optimizer, epoch+1)
    train_loss, train_acc = train(epoch)
    logger.log_value('train_loss', train_loss, epoch)
    logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    if epoch % args.save_epoch == 0:
        checkpoint(epoch)

