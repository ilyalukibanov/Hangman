## This folder contains two naive strategies of guessing words in Hangman game.

The first strategy is to use Marcov Chain transition probabilities derived from consecutive letter probabilities, i.e., what is the probability that 'a' is after 'b'. There are two variations: to compute the transitional probabilities for all words together and to compute the transitional probabilities conditional on the word length.

The second strategy is to use letter cooccurrence in a word, i.e., what is the probability that 'a' is in the word if it contains the letter 'b'. There are two variations as well: to compute the cooccurrence probabilities for all words together and to compute the cooccurrence probabilities conditional on the word length.