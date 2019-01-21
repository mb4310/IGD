import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch

def cleanup(df, standardize=True):
    df.sample(df.shape[0], replace=False).reset_index(drop=True)
    df.columns = [k for k in range(df.shape[1]-1)]+['target']
    for k in df.columns[:-1]:
        df[k] = df[k].astype('float')
        if standardize:
            df[k] = df[k].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    if df.target.dtype == 'object':
        df['target'] = df['target'].apply(lambda x: x.decode('ascii')).astype('int')
    if sorted(df.target.unique()) != list(np.arange(df.target.nunique())):
        new_targs = pd.DataFrame({'target':df.target.unique()}).reset_index()
        df = pd.merge(df, new_targs, left_on='target', right_on='target').drop('target',axis=1).rename(columns={'index':'target'})
    ts = pd.melt(df.reset_index(), id_vars=['index','target'], var_name='time').rename(columns={'index':'id'})
    ts = ts.groupby(['id','time','target']).value.mean().reset_index()
    return df, ts

def fetch_predictions(learner, ts, ct, output=False):
    data = partial_lm_dataset(ct)
    date_range = [k for k in range(1,ct.shape[1]-1)]
    dl = DataLoader(data, batch_size=32)
    learner.data.valid_dl = dl
    x, y = learner.get_preds()
    pct = pd.concat([pd.Series(x[i,:].squeeze().numpy()) for i in range(x.shape[0])], axis=1)
    pct['time'] = date_range
    pts = pd.melt(pct, id_vars='time', var_name = 'id', value_name = 'predicted_value')
    pts.id = pts.id.astype('int')
    df = pd.merge(ts, pts, left_on=['time','id'], right_on=['time', 'id'], how='left').fillna(0)
    return pct, df 

def graph_ts(ts):
    for k in ts.target.unique():
        fig, axes = plt.subplots(figsize=(15,5))
        sns.tsplot(ts[ts.target == k], time='time', unit='id', condition='target', value='value', err_style='unit_traces', ax=axes)    
    fig, axes = plt.subplots(figsize=(15,5))
    sns.tsplot(ts, time='time', unit='id', condition='target', value='value', err_style='unit_traces', ax=axes)
    return None

def graph_predictions(pts):
    ids = np.random.choice(np.arange(pts.id.nunique()), size=20)
    for k in ids:
        piece = pts[pts.id == pts.id.unique()[k]]
        piece.index = piece.time
        fig,ax = plt.subplots()
        piece.value.plot.line()
        piece.predicted_value.plot.line()
        ax.legend()
    return None

def get_cm(clf, val_dl):
    clf.data.valid_dl = val_dl
    x, y = clf.get_preds()
    preds = torch.max(x, dim=1)[1]
    cm = confusion_matrix(y, preds)
    return cm

class cls_ds(Dataset):
    def __init__(self, ts, use_cuda=True):
        self.x = torch.stack([torch.Tensor(ts.iloc[k][:-1].values.astype('float')) for k in range(ts.shape[0])], dim=1)
        self.y = torch.stack([torch.Tensor([ts.iloc[k][-1].astype('int')]).long() for k in range(ts.shape[0])], dim=0)
        self.init_kwargs = None
        self.use_cuda = use_cuda
        
    def __len__(self):
        return self.x.size(1)
    
    def __getitem__(self, idx):
        x,y = self.x[:-1,idx].unsqueeze(1), self.y[idx].squeeze()
        if self.use_cuda:
            x,y = x.cuda(), y.cuda()
        return x,y
