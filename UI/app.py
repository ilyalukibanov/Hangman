import sys
import os
import random
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from flask import Flask, render_template, request, session, redirect, url_for

from modeling.architecture.HangmanTransformer import HangmanTransformer, HTConfig
from modeling.game_simulations.ui_game import game


device = "cuda" if torch.cuda.is_available() else 'cpu'
local_path = ''
last_checkpoint = torch.load('./modeling/models/model_large.pt',map_location=torch.device('cpu'))
model = HangmanTransformer(last_checkpoint['config']) 
model.load_state_dict(last_checkpoint['model'])
model.to(device) 
with open('modeling/data/words_alpha.txt', 'r') as f: 
    data = f.read() 
data = data.splitlines()

app = Flask(__name__)
app.secret_key = "test" + str(random.randint(0,1_000_000))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form['user_name']
        game_mode = request.form['game_mode']
        points_to_win = request.form['n_rounds']
        if user_name and game_mode and points_to_win:
            session['user_name'] = user_name
            session['game_mode'] = game_mode
            session['points_to_win'] = int(points_to_win)
            session['score_human'] = 0
            session['score_transformer'] = 0
            session['turn_number'] = 0
            session['human_words'] = {}
            session['is_round_finished'] = False
            session['transformer_words'] = {}
            session['mistakes_to_lose'] = 8
            return redirect(url_for('play_game'))
    return render_template('index.html')

@app.route('/game', methods=['GET', 'POST'])
def play_game():
    if not session.get('user_name'):
        return redirect(url_for('index'))
    if session['game_mode'] == 'one-sided':
        return transformer_guess()
    else:
        return play_mutual_game()
    
def transformer_guess():
    incorrect_input = False
    INPUT_PLACEHOLDER = "Enter next word..."
    GUESSER = 'transformer'
    if request.method == 'POST' and not session['is_round_finished']:
        user_text = request.form['player-input']
        if user_text.isalpha() and user_text.isascii():
            user_text = user_text.lower()
        else:
            incorrect_input = True
        if incorrect_input:
            return render_template('game.html',
                user_name=session['user_name'],
                score_human=session['score_human'],
                score_transformer=session['score_transformer'],
                is_round_finished=session['is_round_finished'],
                input_placeholder=INPUT_PLACEHOLDER,
                guesser=GUESSER,
                html_result="IE",
                is_game_end=False)
        elif user_text in session['human_words']:
            return render_template('game.html',
                user_name=session['user_name'],
                score_human=session['score_human'],
                score_transformer=session['score_transformer'],
                is_round_finished=session['is_round_finished'],
                input_placeholder=INPUT_PLACEHOLDER,
                guesser=GUESSER,
                html_result="AU",
                is_game_end=False)
        else:
            session['human_words'][user_text] = 1
        history, winner = game(user_text, model)
        html_result = '<h2>Transformer guessed in the following way:</h2>' + ''.join(f"<li>{r}</li>"for r in history)
        if winner == 'human':
            session['score_human'] += 1
            html_result += f"<h3>Round winner: {session['user_name']}</h3>"
        else:
            session['score_transformer'] += 1
            html_result += '<h3>Round winner: Transformer</h3>'
        session['turn_number'] += 1
        session['is_round_finished'] = True
        return render_template('game.html',
            user_name=session['user_name'],
            score_human=session['score_human'],
            score_transformer=session['score_transformer'],
            is_round_finished=session['is_round_finished'],
            input_placeholder=INPUT_PLACEHOLDER,
            guesser=GUESSER,
            html_result=html_result,
            is_game_end=False)
    if request.method == 'POST' and session['is_round_finished']:
        session['is_round_finished'] = False
        if (session['score_transformer'] == session['points_to_win']) or (session['score_human'] == session['points_to_win']):
            if (session['score_transformer'] == session['points_to_win']):
                html_result = "<h1>Transformer won the game!</h1>"
            else:
                html_result = "<h1>You won the game!</h1>"
            return render_template('game.html',
            user_name=session['user_name'],
            score_human=session['score_human'],
            score_transformer=session['score_transformer'],
            html_result=html_result,
            is_game_end=True)
    return render_template('game.html',
        user_name=session['user_name'],
        score_human=session['score_human'],
        score_transformer=session['score_transformer'],
        input_placeholder=INPUT_PLACEHOLDER,
        guesser=GUESSER,
        is_round_finished=session['is_round_finished'],
        is_game_end=False)

def human_guess():
    incorrect_input = False
    INPUT_PLACEHOLDER = 'Enter your next guess'
    GUESSER = 'human'
    if request.method == 'POST' and not session['is_round_finished']:
        user_guess = request.form['player-input']
        if len(user_guess) == 1 and user_guess.isalpha() and user_guess.isascii():
            user_guess = user_guess.lower()
        else:
            incorrect_input = True
            
        if incorrect_input:
            return render_template('game.html',
                user_name=session['user_name'],
                score_human=session['score_human'],
                score_transformer=session['score_transformer'],
                is_round_finished=session['is_round_finished'],
                input_placeholder=INPUT_PLACEHOLDER,
                guesser=GUESSER,
                word_state=session['word_state'],
                html_result="IE",
                n_guesses_remaining=session['mistakes_to_lose']-1-session['n_mistakes'],
                is_game_end=False)
        elif user_guess in session['guessed_letters']:
            return render_template('game.html',
                user_name=session['user_name'],
                score_human=session['score_human'],
                score_transformer=session['score_transformer'],
                is_round_finished=session['is_round_finished'],
                input_placeholder=INPUT_PLACEHOLDER,
                guesser=GUESSER,
                word_state=session['word_state'],
                n_guesses_remaining=session['mistakes_to_lose']-1-session['n_mistakes'],
                html_result=session['html_human'],
                is_letter_error=True,
                is_game_end=False)
        else:
            session['guessed_letters'][user_guess] = 1

        if user_guess in session['word']:
            session['correct_guesses'][user_guess] = 1
            session['html_human'] += f'<li>correct guess: {user_guess}</li>'
            word_state = re.sub(rf"[^{','.join(session['correct_guesses'])}]", '_', session['word'])
            session['word_state'] = '\u2009'.join(word_state)
            session['html_human'] += f'<li>guessed word part: {session["word_state"]}</li>'
            if word_state == session['word']:
                session['is_round_finished'] = True
                session['html_human'] += f"<h3>Round winner: {session['user_name']}</h3>"
                session['turn_number'] += 1
                session['score_human'] += 1
        else:
            session['html_human'] += f'<li>wrong guess: {user_guess}</li>'
            session['n_mistakes'] += 1
        
        if session['n_mistakes'] == session['mistakes_to_lose']:
            session['is_round_finished'] = True
            session['html_human'] += f"<li>The word was: {session['word']}</li>"
            session['html_human'] += f"<h3>Round winner: Transformer</h3>"
            session['turn_number'] += 1
            session['score_transformer'] += 1

        
        return render_template('game.html',
            user_name=session['user_name'],
            score_human=session['score_human'],
            score_transformer=session['score_transformer'],
            is_round_finished=session['is_round_finished'],
            input_placeholder=INPUT_PLACEHOLDER,
            guesser=GUESSER,
            word_state=session['word_state'],
            html_result=session['html_human'],
            n_guesses_remaining=session['mistakes_to_lose']-1-session['n_mistakes'],
            is_game_end=False)
    if request.method == 'POST' and session['is_round_finished']:
        if (session['score_transformer'] == session['points_to_win']) or (session['score_human'] == session['points_to_win']):
            if (session['score_transformer'] == session['points_to_win']):
                html_result = "<h1>Transformer won the game!</h1>"
            else:
                html_result = "<h1>You won the game!</h1>"
            return render_template('game.html',
            user_name=session['user_name'],
            score_human=session['score_human'],
            score_transformer=session['score_transformer'],
            html_result=html_result,
            is_game_end=True)
        global data
        session['is_round_finished'] = False
        session['guessed_letters'] = {}
        session['correct_guesses'] = {}
        session['html_human'] = ''
        session['n_mistakes'] = 0
        session['word'] = data[random.randint(0,len(data))]
        while session['word'] in session['transformer_words']:
            session['word'] = data[random.randint(0,len(data))]
        session['transformer_words']['word'] = 1
        session['word_state'] = '\u2009'.join('_'*len(session['word']))

    return render_template('game.html',
        user_name=session['user_name'],
        score_human=session['score_human'],
        score_transformer=session['score_transformer'],
        input_placeholder=INPUT_PLACEHOLDER,
        guesser=GUESSER,
        word_state=session['word_state'],
        is_round_finished=session['is_round_finished'],
        n_guesses_remaining=session['mistakes_to_lose']-1-session['n_mistakes'],
        is_game_end=False)


def play_mutual_game():
    if session['turn_number'] % 2 == 0:
        return transformer_guess()
    else:
        return human_guess()

if __name__ == '__main__':
    app.run(debug=True)