import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import copy; import time; import pdb
import torch

class Smoother():
    def __init__(self, beta=0.95):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.vals = []

    def add_value(self, val):
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1-self.beta)*val
        self.vals.append(self.mov_avg/(1-self.beta**self.n))

    def process(self,array):
        for item in array:
            self.add_value(item)
        return self.vals

    def reset(self):
        self.n, self.mov_avg, self.vals = 0,0,[]

class Stepper():
    def __init__(self, opt):
        self.it = 0
        self.opt = opt
        self.nits = 1

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
    
    @staticmethod
    def cosine_anneal(pct, max_val, min_val):
        return min_val + (max_val - min_val) / 2 *(1+np.cos(np.pi * pct))
    
    @staticmethod
    def exp_anneal(pct, start, stop):
        return start * (stop/start)**pct
    
    @staticmethod
    def linear_anneal(pct, start, stop):
        return (1-pct)*start + pct*stop
    
class OneCycle(Stepper):
    def __init__(self, opt, nits=1, max_lr=1e-4, momentums=[0.85,0.95], div=25, pct_start=0.3):
        super(OneCycle, self).__init__(opt)
        self.nits = nits
        self.max_lr = max_lr
        self.momentums = momentums
        self.div = div
        self.pct_start = pct_start
        self.phase = 0
        self.switch = int(pct_start * nits)
    
    def step(self):
        self.opt.step()
        self.it += 1
        if self.phase == 0: 
            pct = self.it / (self.nits * self.pct_start)
            new_lr = self.linear_anneal(pct, self.max_lr/div, self.max_lr)
            new_mom = self.linear_anneal(pct, self.momentums[0], self.momentums[1])
            for group in self.opt.param_groups:
                group['lr'] = new_lr
                if 'betas' in group.keys():
                    group['betas'] = (new_mom, group['betas'][1])
                else:
                    group['momentum'] = new_mom
            if self.it > self.switch:
                self.phase += 1
                self.it = 0
        
        else: 
            pct = self.it / (self.nit * (1-self.pct_start))
            new_lr = self.cosine_anneal(pct, self.max_lr, self.max_lr * 1e-5)
            new_mom = self.cosine_anneal(pct, self.momentums[1], self.momentums[0])
            for group in self.opt.param_groups:
                group['lr'] = new_lr
                group['beta'] = (new_mom, group['betas'][1])

class LearningRateFinder(Stepper):
    def __init__(self, opt, nits=1, min_lr=1e-6, max_lr=1e1):
        super(LearningRateFinder, self).__init__(opt)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.nits = nits
        for group in self.opt.param_groups:
            group['lr'] = min_lr
    
    def step(self):
        self.opt.step()
        self.it+=1 
        new_lr = self.exp_anneal(self.it / self.nits, self.min_lr, self.max_lr)
        for group in self.opt.param_groups:
            group['lr'] = new_lr
    

def train(model, criterion, stepper, dataloaders, num_epochs=1):
    start = time.time()
    dataset_sizes = {'train':len(dataloaders['train'].dataset), 'val':len(dataloaders['val'].dataset)}
    tr_losses = []
    lrs = []
    epoch_losses = {'train': [], 'val': []}
    epoch_accs = {'train': [], 'val': []}
    stepper.nits = num_epochs * len(dataloaders['train'].dataset) / (dataloaders['train'].batch_size)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_losses = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                stepper.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    stepper.step()
                    tr_losses.append(loss.item())
                    lrs.append(stepper.opt.param_groups[-1]['lr'])
                
                running_losses += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_losses / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_losses[phase].append(epoch_loss)
            epoch_accs[phase].append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    smoother = Smoother()
    tr_losses = smoother.process(tr_losses)
    tr_history = pd.DataFrame({'tr_loss':tr_losses, 'learning_rate':lrs}).reset_index().rename(columns={'index':'iteration'})
    epoch_history = pd.DataFrame(OrderedDict({'tr_loss': epoch_losses['train'], 'tr_acc' : epoch_accs['train'], 
    'val_loss':epoch_losses['val'], 'val_acc':epoch_accs['val']})).reset_index().rename(columns={'index':'epoch'})

    elapsed_time = time.time() - start
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))    
    print(elapsed_time)
    return model, (tr_history, epoch_history)


                    
                    


    