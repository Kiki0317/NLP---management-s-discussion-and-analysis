# NLP---management-s-discussion-and-analysis
we download 10-k&amp;10-q from SEC and extract MD&amp;A chapter from each file. Then we clean the extracted text: remove stop words, punctuations, tags, numbers......
use nltk and master dictionary to tokenize and filter words through adding weights (TF-IDF method)
Then we will train model based on result from last step, Naive Bayers, Logistic Regression
