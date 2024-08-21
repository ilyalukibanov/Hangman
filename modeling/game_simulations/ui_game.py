import torch
from torch.nn import functional as F 
import re

chars = list(".abcdefghijklmnopqrstuvwxyz_") # all the possible characters 
stoi = {ch:i for i,ch in enumerate(chars)} 
itos = {i:s for s,i in stoi.items()} # inverse mapping 
device = "cpu" 


def get_transition_order(model_input, model):
    """
    Predict likelihoods for each letter given the model state.
    Return the ordered most likely letters.
    """
    logits, _ = model(model_input,return_logits=True) 
    logits = logits.mean(1).exp() 
    probs = F.softmax(logits, dim=-1) 
    transition_order = torch.argsort(probs, descending=True) 
    return transition_order[0] 

def game(word, model): 
    """
    Simulate a Hangman game
    """
    guess_position = 1 
    correct_letters_needed = len(set(word)) 
    model_input = torch.zeros((1,32),dtype=torch.int32, requires_grad=False, device=device) 
    model_input[:,:len(word)] = 27 
    transition_order = get_transition_order(model_input, model) 
    guess = itos[transition_order[0].item()] 
    guesses = set(['.']) 
    n_mistakes = 0 
    n_correct = 0 
    history = []
    while n_mistakes < 8 and n_correct < correct_letters_needed: 
        guesses.update(guess) 
        if guess in word: 
            n_correct += 1 
            history.append(f'correct guess: {guess}')
            word_model = re.sub(rf"[^{','.join(guesses)}]", '_', word)
            history.append(f"guessed word part: {word_model}")
            guess_position = 0 
            correct_guess_positions = [i for i in range(len(word)) if word.startswith(guess, i)] 
            model_input[:,correct_guess_positions] = stoi[guess] 
            transition_order = get_transition_order(model_input, model) 
            guess = itos[transition_order[guess_position].item()] 
            while (guess in guesses): 
                guess_position += 1 
                guess = itos[transition_order[guess_position].item()] 
        else: 
            history.append(f'wrong guess: {guess}')
            n_mistakes += 1 
            guess_position += 1 
            guess = itos[transition_order[guess_position].item()] 
            while (guess in guesses): 
                guess_position += 1 
                guess = itos[transition_order[guess_position].item()] 
    if (n_correct == correct_letters_needed): 
        winner = 'transformer'
    else:
        winner = 'human'
    return (history, winner)