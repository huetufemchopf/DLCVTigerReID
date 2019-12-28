import os
import torch

import parser1
import models
from featureextractor import FeatureExtractor, Model
import data
from evaluate import get_acc
import numpy as np
import torch.nn as nn
from triplet_loss import TripletLoss, get_dist, get_dist_local

from tensorboardX import SummaryWriter
#from test import evaluate


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    args = parser1.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    gallery_loader = torch.utils.data.DataLoader(data.DATA(args, mode='gallery'),
                                             batch_size=40,
                                             num_workers=args.workers,
                                             shuffle=False)
    query_loader = torch.utils.data.DataLoader(data.DATA(args, mode='query'),
                                             batch_size=args.test_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    model = Model()
    model.cuda()  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    t_loss = TripletLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    iters = 0
    best_acc = 0
    long = len(train_loader)
    #print_num = int(long/10)
    print_num = 6

    print('===> start training ...')
    for epoch in range(1, args.epoch + 1):

        model.train()
        avg_loss = 0

        for idx, (imgs, cls) in enumerate(train_loader):
            all_img = []
            all_labels = []

            for img_list, lab in zip(imgs, cls):
                for i in range(len(img_list)):
                    all_img.append(img_list[i])
                    all_labels.append(lab)

            ''' move data to gpu '''
            all_img = torch.stack(all_img)
            all_labels = torch.stack(all_labels)
            all_img, all_labels = all_img.cuda(), all_labels.cuda()

            ''' forward path '''
            global_f, local_f, classes = model(all_img)

            ''' compute loss, backpropagation, update parameters '''
            # Global losses
            dist_1_g, dist_2_g = get_dist(global_f, all_labels)
            loss_g = t_loss(dist_1_g, dist_2_g)

            # Local losses
            dist_1_l, dist_2_l = get_dist_local(local_f, all_labels)
            loss_l = t_loss(dist_1_l, dist_2_l)

            loss_c = criterion(classes, all_labels)

            loss = loss_g + loss_l + loss_c

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters
            avg_loss += loss
            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)

            if (idx+1) % print_num == 0:
                print('epoch: %d/%d, [iter: %d / %d], Mean loss: %f' \
                      % (epoch, args.epoch, idx + 1, long, avg_loss/print_num))
                avg_loss = 0

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            model.eval()
            acc = get_acc(model, query_loader, gallery_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc


        ''' save model '''
        #save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
