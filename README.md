# SentimentAdaptNLP
 Cross-domain sentiment analysis using NLP techniques, COMP 6480

The nlp_project.ipynb contains data exploration, preparation and preprocessing. 
The scripts folder containing all python scripts that used in training the model via the job scheduler on the server. 

* tokenize_and_save.py - code used for tokenizing both Electronics and Movies datasets.
* train_bert.py - code for training the BERT model for the Electronics dataset and testing on the Movies dataset (cross-domain).
* train_bert_movies.py - code for training another model for a within-domain baseline comparison. 
