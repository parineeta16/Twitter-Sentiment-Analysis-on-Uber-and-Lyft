# Twitter-Sentiment-Analysis-on-Uber-and-Lyft

The project uses three datasets: Uber Dataset, Lyft Dataset, and Training Dataset. \
\
Uber and Lyft datasets are created using the Tweepy API. \
To use Tweepy API and fetch tweets from Twitter, we must create a Twitter Development Account. Once the account is approved, we get the credentials to access data from Twitter. \
\
The training dataset contains labelled airline sentiment data that has tweets about different airlines from the past 5 years.

- For data pre-processing, NLP is used. 
- Multinomial Naive Bias, Random Forest, and Support Vector Machine are applied to perform sentiment analysis. 
- An ensemble model called a voting classifier is used to improve performance. 
- The model is pickled into a file and then used on Uber and Lyft datasets to save prediction time. 
- Word Clouds are plotted to obtain most common words used in positive and negative reviews of both the datasets. 

Visualizations and analysis are made based on the model output and Word Cloud.

![plot](https://github.com/parineeta16/Twitter-Sentiment-Analysis-on-Uber-and-Lyft/blob/master/Uber.jpg)
![plot](https://github.com/parineeta16/Twitter-Sentiment-Analysis-on-Uber-and-Lyft/blob/master/Lyft.jpg)

Difficulties:

1) Many of the tweets were sarcastic and
irrelevant to the ride-sharing companies.

2) Applying decision tree algorithms like Random
Forest was quite a challenge since it consumed
memory, CPU power, and would take days to
run making it difficult to fine-tune.

