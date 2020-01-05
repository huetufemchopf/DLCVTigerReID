import pandas as pd
import parser1
import os
args = parser1.arg_parse()


def readfilenames(folder):
    names = []
    for i in os.listdir(os.path.join(folder, 'imgs')):
        names.append(i)
    df_files = pd.DataFrame(names, columns=['filename'])
    return df_files


def getunlabeled(folder, df_files):

    files = readfilenames(folder)

    df_train = pd.read_csv(os.path.join(folder, 'train.csv'), header=None, names= ['ID', 'filename'])
    df_query = pd.read_csv(os.path.join(folder, 'query.csv'), header=None, names= ['ID', 'filename'])
    df_gallery = pd.read_csv(os.path.join(folder, 'gallery.csv'), header=None, names= ['ID', 'filename'])


    df_files = df_files[~df_files.filename.isin(df_query.filename)]
    df_files = df_files[~df_files.filename.isin(df_train.filename)]
    df_files = df_files[~df_files.filename.isin(df_gallery.filename)]

    df_files.to_csv(os.path.join(folder, 'train_unlabeled.csv'), header=False, index=False)

# def getlabelsfromfulldata(folder):
#     df_trainfull = pd.read_csv(os.path.join(folder, 'reid_list_train.csv'), header=None, names=['ID', 'filename'])
#     df_query = pd.read_csv(os.path.join(folder, 'query.csv'), header=None, names=['ID', 'filename'])
#     df_gallery = pd.read_csv(os.path.join(folder, 'gallery.csv'), header=None, names= ['ID', 'filename'])
#
#     df_trainfull = df_trainfull[~df_trainfull.filename.isin(df_query.filename)]
#     df_trainfull = df_trainfull[~df_trainfull.filename.isin(df_gallery.filename)]
#
#     # df_files.to_csv(os.path.join(folder, 'train_fulldata.csv'), header=False, index=False)
#
#     print(df_trainfull.describe)


files = readfilenames(args.data_dir)
getunlabeled(args.data_dir, files)

# getlabelsfromfulldata(args.data_dir)