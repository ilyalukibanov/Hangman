import torch 
from architecture.HangmanTransformer import HangmanTransformer, HTConfig
from game_simulations.training_game import test_game

device = "cuda" if torch.cuda.is_available() else 'cpu'
last_checkpoint = torch.load('./models/model_large.pt',map_location=torch.device(device))
model = HangmanTransformer(last_checkpoint['config']) 
model.load_state_dict(last_checkpoint['model'])
model.to(device) 

with open('data/words_alpha.txt', 'r') as f: 
    data = f.read() 
words = data.splitlines()

test_game(words, model, 100)