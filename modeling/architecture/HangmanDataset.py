import torch
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader

class WordDataset(Dataset): 
 
    def __init__(self, x, y, chars, input_length): 
        self.x = x 
        self.y = y 
        self.chars = chars 
        self.input_length = input_length 
        self.stoi = {ch:i for i,ch in enumerate(chars)} 
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping 
 
    def __len__(self): 
        return len(self.x) 
 
    def contains(self, word): 
        return word in self.words 
 
    def encode(self, word): 
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.uint8) 
        return ix 
 
    def decode(self, ix): 
        word = ''.join(self.itos[i] for i in ix) 
        return word 
 
    def __getitem__(self, idx): 
        x = self.encode(self.x[idx]) 
        y = self.y[idx] 
        return x, y 
     
def load_dataset(): 
 
    with open(f'data/x.txt', 'r') as f: 
        train_x = f.read() 
    train_x = train_x.splitlines() 
    train_x = [word.ljust(32, '.') for word in train_x] 
     
    train_y = torch.load(f'data/y.pt') 
     
    chars = list(".abcdefghijklmnopqrstuvwxyz_") # all the possible characters 
    input_length = len(chars) 
 
    # wrap in dataset objects 
    train_dataset = WordDataset(train_x, train_y, chars, input_length) 
    
    return train_dataset 
 
class InfiniteDataLoader: 

    def __init__(self, dataset, **kwargs): 
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10)) 
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs) 
        self.data_iter = iter(self.train_loader) 
 
    def next(self): 
        try: 
            batch = next(self.data_iter) 
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never) 
            self.data_iter = iter(self.train_loader) 
            batch = next(self.data_iter) 
        return batch