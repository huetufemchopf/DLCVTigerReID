import os
import argparse

import pandas as pd
import torch
from numpy import array

from sklearn.metrics import accuracy_score

def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)

def evaluate(query, gallery, pred):
    
    assert query.shape[0] == pred.shape[0]
    
    pred = pred.squeeze()
    qurey_id = query[:, 0].tolist()
    pred_id  = []
    gallery_dic = dict(zip(gallery[:,1], gallery[:,0]))
    
    for p in pred:
        pred_id.append(gallery_dic[p])
    
    return accuracy_score(qurey_id, pred_id)*100


def get_acc(model, query, gallery):
    with torch.no_grad():
        q_arr = []
        q_name = 0
        loss = torch.nn.MSELoss()
        for g_img, g_name in gallery:
            g_img = g_img.cuda()
            q_name = g_name
            g_out = model(g_img)
            for idx, (imgs, _) in enumerate(query):
                imgs = imgs.cuda()
                output = model(imgs)
                q_arr.append(output)

        q_out = torch.cat(q_arr)
        preds = []
        for q in q_out:
            min_e = 10000
            pred = ''
            for g, n in zip(g_out, q_name):
                error = loss(q, g)
                if error < min_e:
                    min_e = error
                    pred = n
            preds.append(pred)
    q_csv = read_csv('dataset/query.csv')
    g_csv = read_csv('dataset/gallery.csv')
    acc = evaluate(q_csv.values, g_csv.values, array(preds))
    return acc

if __name__ == '__main__':

    '''argument parser'''
    parser = argparse.ArgumentParser(description='Code to evaluate DLCV final challenge 1')
    parser.add_argument('--query', type=str, help='path to query.csv') 
    parser.add_argument('--gallery', type=str, help='path to gallery.csv')
    parser.add_argument('--pred', type=str, help='path to your predicted csv file (e.g. predict.csv)')
    args = parser.parse_args()

    ''' read csv files '''
    query = read_csv('dataset/query.csv')
    gallery = read_csv('dataset/gallery.csv')
    pred = read_csv('prueba.csv')
    

    rank1 = evaluate(query.values, gallery.values, pred.values)
    
    print('===> rank1: {}%'.format(rank1))

