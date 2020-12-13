# Text-Mining
## Bag of Authors - Guess Who wrote this?

The premise is simple, given a corpora of a few texts :books: by some authors, can you predict who wrote what? :sweat_smile:

Two levels of difficulty, a larger text of 1000 words or with only 500 words...

The game is on, and since the corpora for each author is very different we balance the data by collecting the same number of samples from each author by checking how many texts each author has, and deciding how many to collect in total.
Several different samples can then be drawn from the texts since the idea is to capture the style of the author and not a particular subject.
We then proceed to clean each sample and thoroughly process them removing and replacing parts specific characters.
We then continue experimenting with TF-IDF and Bag of Words , using n-grams and features created while normalizaing the samples.

We experiment using K-nearest neighbors, Multinomial Logistic Regression, a simple Feed Forward Neural network with a hidden layer.

We then compare results obtained with the different composition on models, on our test and deliver a model to predict a hidden test for evaluation.

The full Report is also available for consideration.
Final grade was 
