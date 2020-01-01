from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Parser for TigerReID by team Wondaba for course DLCV')

    '''Enviroment parameter'''
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--random_seed', type=int, default=200)

    '''Datasets parameters'''
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                       help="number of data loading workers (default: 4)")

    '''Training parameters'''
    parser.add_argument('--epoch', default=300, type=int,
                        help="num of total epochs")
    parser.add_argument('--train_batch', default=30, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int,
                        help="query batch size")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="num of epochs a val is run")
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--label_group', default=3, type=int,
                        help="Number of grouped images with the same label")
    parser.add_argument('--global_mult', default=2, type=float,
                        help="Multiplyer of global losses")
    parser.add_argument('--local_mult', default=1, type=float,
                        help="Multiplyer of local losses")
    parser.add_argument('--class_mult', default=2, type=float,
                        help="Multiplyer of class losses")
    parser.add_argument('--vertical_mult', default=1, type=float,
                        help="Multiplyer of vertical losses")


    '''Optimizer params'''
    parser.add_argument('--lr', default=0.00005, type=float,
                       help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help="initial Weight Decay")

    '''Optional Improvements'''
    parser.add_argument('--lr_change', default=True, type=bool,
                        help="Change learning rate or not")
    parser.add_argument('--grouping', default=False, type=bool,
                        help="Groups the images in same categories or not")

    '''Resume trained model for test'''
    parser.add_argument('--resume', type=str, default='log',
                      help="path to the trained model")

    args = parser.parse_args()

    return args
