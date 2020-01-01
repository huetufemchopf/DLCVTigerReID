import torch
import data
import data_ng


def loader(args):
    if args.grouping:
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
