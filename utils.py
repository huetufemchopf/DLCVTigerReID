import torch
import data
import data_ng
import torch.nn as nn


def loader(args):
    if args.grouping:
        if args.random_sampling:
            train_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='train'),
                                                       batch_size=args.train_batch,
                                                       num_workers=args.workers,
                                                       shuffle=True)
            gallery_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='gallery'),
                                                         batch_size=40,
                                                         num_workers=args.workers,
                                                         shuffle=False)
            query_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='query'),
                                                       batch_size=args.test_batch,
                                                       num_workers=args.workers,
                                                       shuffle=False)

        else:
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
    else:
        train_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='train'),
                                                   batch_size=args.train_batch,
                                                   num_workers=args.workers,
                                                   shuffle=True)
        gallery_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='gallery'),
                                                     batch_size=40,
                                                     num_workers=args.workers,
                                                     shuffle=False)
        query_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='query'),
                                                   batch_size=args.test_batch,
                                                   num_workers=args.workers,
                                                   shuffle=False)

    return train_loader, gallery_loader, query_loader


def group_imgs(imgs, cls):
    all_img = []
    all_labels = []

    for img_list, lab in zip(imgs, cls):
        for i in range(len(img_list)):
            all_img.append(img_list[i])
            all_labels.append(lab)

    all_img = torch.stack(all_img).cuda()
    all_labels = torch.stack(all_labels)
    return all_img, all_labels


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss