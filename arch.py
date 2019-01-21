import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn_dropout(nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p
    
    @staticmethod
    def dropout_mask(x, sz, p):
        return x.new(*sz).bernoulli_(1-p).div_(1-p)
    
    def forward(self, x):
        if not self.training or self.p == 0: return x
        m = self.dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x*m
    
class lstm_cls(nn.Module):
    def __init__(self, nf, nh, ncl, n_classes=2, p=0.3, pool=True):
        super(lstm_cls, self).__init__()
        self.pool = pool
        self.encode = nn.LSTM(nf, nh, batch_first=True)
        self.cores = nn.ModuleList([nn.LSTM(nh, nh, batch_first=True) for l in range(ncl)])
        self.drops = nn.ModuleList([rnn_dropout(p) for l in range(ncl)])
        if pool: 
            self.decode = nn.Linear(3*nh, int(nh/2))
            self.out = nn.Linear(int(nh/2), n_classes)
        else:
            self.decode = nn.Linear(nh, int(nh/2))
            self.out = nn.Linear(int(nh/2), n_classes)
        
    def Pool(self, x, is_max=False):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(0,2,1),1).view(x.size(0),-1)

    def forward(self, x): 
        x, _ = self.encode(x)
        for core, drop in zip(self.cores, self.drops):
            x = drop(core(x)[0])
        if self.pool:
            maxpool = self.Pool(x, is_max=True)
            avgpool = self.Pool(x)
            x = torch.cat([x[:,-1,:], maxpool, avgpool], dim=1)
        else:
            x = x[:,-1,:]
        output = self.out(F.relu(self.decode(x), inplace=True))
        return output