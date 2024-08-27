# Naive-Bayes-for-sentiment-analysis-on-tweets.
To run this code, you first need to create a text file named input.txt. Write a tweet in this file to get the desired output.

This code implements a basic sentiment analysis system using the Naive Bayes algorithm. The process begins with text preprocessing, where tweets are tokenized, converted to lowercase, stripped of punctuation, and filtered for stopwords. This step is crucial as it cleans and standardizes the data, making it easier for the machine learning model to process. The preprocess_tweet function handles this task by leveraging NLTK’s word_tokenize and stopwords modules, ensuring that only relevant words remain in the dataset for analysis.

The core of the system is the Naive Bayes model, which is trained using labeled tweets—examples of text with predefined sentiment labels (positive or negative). The train_nb_model function calculates the frequency of words within positive and negative tweets and estimates the prior probabilities for each sentiment class. This frequency distribution allows the model to learn which words are more likely to appear in positive or negative contexts. When new tweets are processed, the make_predictions function uses these learned frequencies to calculate the likelihood of each tweet being positive or negative, ultimately predicting the sentiment based on these probabilities.

Finally, the code includes several analyses to evaluate and interpret the model's performance. This includes error analysis to measure the model’s accuracy, the identification of the most common positive and negative words, and an examination of the most informative features—words that are strongly associated with either positive or negative sentiments. By identifying these features, the code provides insight into which words are most influential in determining sentiment, offering a deeper understanding of the model's decision-making process.






