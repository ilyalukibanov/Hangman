import os 
 
import torch 
from torch.nn import functional as F 

from architecture.HangmanTransformer import HangmanTransformer, HTConfig
from architecture.HangmanDataset import load_dataset, InfiniteDataLoader
from game_simulations.training_game import test_game

device = "cuda" if torch.cuda.is_available() else 'cpu'
model_name = 'large'
torch.set_float32_matmul_precision('high')

with open(f'data/words_alpha.txt', 'r') as f: 
    data = f.read() 
words = data.splitlines() 
 
model = HangmanTransformer(config=HTConfig()) 
model.to(device) 
i = 0 

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.99), eps=1e-8)
train_dataset = load_dataset()
batch_loader = InfiniteDataLoader(train_dataset, batch_size=768, pin_memory=True, num_workers=1)

while True: 
    batch = batch_loader.next() 
    batch = [t.to(device) for t in batch] 
    x, y = batch 
    optimizer.zero_grad() 
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        _, loss = model(x.to(torch.int32), y, return_logits=False) 
    loss.backward() 
    optimizer.step() 
    if device == "cuda": 
        torch.cuda.synchronize() # wait for the GPU to finish work 
    if i % 100 == 0: 
        print(loss) 
    if (i > 0) and (i % 5000 == 0): 
        checkpoint_path = os.path.join(f'models/{model_name}/checkpoints/', f"model_{i:07d}.pt") 
        checkpoint = { 
                    'model': model.state_dict(), 
                    'config': model.config, 
                    'step': i 
                } 
        torch.save(checkpoint, checkpoint_path) 
        test_game(words, model) 
    i += 1