# Text-Mining
## Bag of Authors - Guess Who wrote this?

The premise is simple: Given a [text corpora](https://en.wikipedia.org/wiki/List_of_text_corpora) of a few texts :books: by some authors, can you predict who wrote what? :sweat_smile:  
  
Two levels of difficulty, a larger text of 1000 words or harder given only 500 words...  
  
The game is on! And since the corpora for each author is very different, we balance the data by collecting the same number of samples from each author by checking how many texts each author has, and deciding how many to collect in total.  
Several different samples can then be drawn from the texts since the idea is to capture the style of the author and not a particular subject.  
We then proceed to clean each sample and thoroughly process them applying [tokenizers](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization), [stemmers](https://en.wikipedia.org/wiki/Stemming) and [lemmatizers](https://en.wikipedia.org/wiki/Lemmatisation) and creating additional features for each author.  
We then continue experimenting with [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) , using [n-grams](https://en.wikipedia.org/wiki/N-gram) and features created while normalizing the samples.  
  
We experiment using K-nearest neighbors, Multinomial Logistic Regression and a simple Feed Forward Neural network with a hidden layer, using Keras.  
  
We then compare results obtained with the different composition on models on our test and finally deliver a model to predict a hidden test for evaluation, and given the results from test we were pretty confident on a good outcome! :v:  
  
Check the report for details, but here goes a confusion matrix on the test:  
![Val_PerfectWW](https://user-images.githubusercontent.com/49924571/102026894-8de74780-3d98-11eb-9213-f96ffed2d04d.png)  
  
The full Report is also available for consideration.  
Final grade was 19/20.  
  
It was fun to play around with NLP! :wink:  

