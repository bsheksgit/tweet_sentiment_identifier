This is a tweet sentiment classifier created using the Bag of words method and XGBoost.

Tweets were cleaned first by removing the special characters like #$%&*@ etc. Then these cleaned tweets were then lemmatized to extract the lemma from the words. The lemmatized tweets were fed to the count vectorizer to create features from the tweets. 
Next these features were fed to the XGboost classifier to create a model which can identify the sentiment of the given tweet.