# Naive-Bayes-for-sentiment-analysis-on-tweets.
Input Tweets File: This file, named tweet.txt, should contain tweets for which you want to perform sentiment analysis. Each line in the file represents a single tweet. If you choose to input a tweet directly, it will be appended to the list of tweets from this file. The script reads this file, preprocesses each tweet to remove stopwords and punctuation, and then classifies the sentiment using the custom Naive Bayes model.

The datasets provided are two distinct CSV files named `twitter_training.csv` and `twitter_validation.csv`, each containing different sets of data related to sentiment analysis tasks. Both datasets consist of text data extracted from tweets or similar text sources, with accompanying labels that indicate sentiment or relevance. 

The first dataset, `twitter_training.csv`, is a larger file comprising 74,681 rows and 4 columns. The columns appear to represent identifiers or category labels, sentiment classification, and the actual text content. The text entries in this dataset are associated with a particular sentiment, such as "Positive," and relate to specific topics or categories. For example, many text entries seem to be centered around a theme like "Borderlands," a popular video game. This dataset is
likely used in training a machine learning model to recognize and classify sentiments expressed in text, a common task in natural language processing (NLP). 

On the other hand, the `twitter_validation.csv` dataset is smaller, with 999 rows and 4 columns. Like the training dataset, it contains identifiers, category labels, and text content. However, this dataset also includes different topics, such as "Facebook," "Amazon," and "Google," with associated labels like "Neutral,"  Negative," and "Irrelevant." This variety suggests that the validation dataset is used to test the model's performance across a broader range of categories  and ntiments, ensuring that the model generalizes well to different kinds of text inputs.

The code implements a sentiment analysis pipeline using both a custom Naive Bayes model and a more standard approach using the scikit-learn library. Here's a step-by-step explanation of the implementation:
•	Importing Libraries:
i.	Natural Language Toolkit (NLTK) is used for text processing tasks such as tokenization and removing stopwords.
ii.	sklearn: Scikit-learn is used for machine learning tasks, including training the Naive Bayes model and evaluating its performance.
iii.	pandas and numpy: These are used for data manipulation and numerical computations.

•	Custom Naive Bayes Model Functions: The code defines three functions to implement a custom Naive Bayes model:
i.	train_nb_model: This function calculates word frequencies based on the labels (positive or negative) and computes the prior probabilities for each class.
ii.	preprocess_tweet: This function preprocesses each tweet by converting it to lowercase, tokenizing it into words, removing non-alphabetic characters, and filtering out common English stopwords.
iii.	make_predictions: This function uses the word frequencies and prior probabilities calculated during training to predict the sentiment (positive or negative) of new tweets. 

•	Training the Custom Naive Bayes Model: A small set of labeled tweets is manually provided to train the custom Naive Bayes model. Each tweet is preprocessed using the `preprocess_tweet` function, and the model is trained using the `train_nb_model` function.

•	Input Tweets and Prediction: The code then loads input tweets from a file named `tweet.txt`. Optionally, the user can also input a tweet directly. These tweets are preprocessed and fed into the `make_predictions` function, which predicts the sentiment of each tweet using the custom Naive Bayes model.

•	Loading and Preparing Datasets for Scikit-learn: The code proceeds to load the training and validation datasets from two CSV files, `twitter_training.csv` and `twitter_validation.csv`. It renames the columns for clarity and filters out irrelevant and neutral sentiments, keeping only positive and negative ones. Missing values are handled by dropping rows where the tweet content is `NaN`.

•	Text Vectorization:The tweets in the training dataset are transformed into a matrix of token counts using the `CountVectorizer` from scikit-learn. This converts the text data into numerical form, which is required for model training.

•	Training the Scikit-learn Naive Bayes Model:A Naive Bayes model (specifically, a Multinomial Naive Bayes model) is trained using the token counts from the training data. This model is then used to predict the sentiments of tweets in the validation dataset.

•	Model Evaluation:The predictions made on the validation dataset are compared with the actual labels to evaluate the model's performance. The accuracy, classification report, and confusion matrix are computed and printed to assess how well the model performs on unseen data.

•	NLTK Resource Download: Finally, the necessary NLTK resources for tokenization and stopwords are downloaded to ensure that the text preprocessing functions work correctly.

In summary, the code implements two approaches to sentiment analysis: a custom Naive Bayes model and a more conventional scikit-learn-based model. The custom model is trained on a small, manually provided dataset, while the scikit-learn model is trained and evaluated on larger datasets, with performance metrics computed to measure its effectiveness.






