# NLP---management-s-discussion-and-analysis
Project introduction:

step1: we download 10-k&amp;10-q from SEC and extract MD&amp;A chapter from each file. Then we clean the extracted text: remove stop words, punctuations, tags, numbers......

step2: use nltk and master dictionary to tokenize and filter words through adding weights (TF-IDF method)

step3：Then we will train model based on result from last step, Naive Bayers, Logistic Regression
