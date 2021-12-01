# Should You Buy This Game?? 
Team: Daniel Mata, YingHsuan Lo

[Project Repository](https://github.com/loriylo/video_game__review_NLP)

[Increment 1 video link](https://drive.google.com/file/d/1Z0BsvbKfvIc-mNvAfTlEwlAd-wfwtoPA/view?usp=sharing)

## Motivation

With video games being in abundance and game console technology getting better and better, video game journalists are constantly writing game reviews about the latest video games at any given moment. Sometimes, readers may just want a quick opinion on whether or not to buy a video game. They may want to get the gist of how a video game journalist has described a certain video game. Additionally, some readers may be visually impaired or have some other form of disability that requires some audio component when browsing the web. Or some may just simply want to pick the game when they are doing chores, driving, or other routines-- treat a purchase choice more like an experience of listening to an audio book.

As such, we choose to focus on text summarization, text classification, and speech synthesis in order to help facilitate video selection for buyers as well as to elevate market penetration for sellers. The main aspect of our project is the text classification, we would like to see if we can improve upon past work using the Steam Reviews data (more information provided in sections below). We want to provide a quick way for users to decide on whether or not to buy a video game.

A transparent and automatic analyzing and classifying system is needed, yet few seriously consider it in academia. For future market analysis research and for more people to accurately pick the right video games, we propose a system which performs video game review text collection from IGN video game website, text summarization and classification on the review, and finally performs text-to-speech. The result will be incorporated showing a game’s summary, visualized recommendation, and speech synthesis.

## Significance

Video games have become more and more influential in every endeavor. From school educational platforms to museum programs, video games ignite interest in many. An interesting and appropriate video game can be utilized in various ways ranging from relaxation( and de-stressing), making new friends, and even advanced topics such as reinforcement learning. As a result, a functional system that can be used to quickly assess a user’s decision on whether or not to buy a video game can be very helpful. In this design, we especially add speech synthesis as an additional component to visualization elements in order to better assist those who suffer from any visually-impairing disability.

## Dataset:
### Steam website and review dataset
Dataset 2 : [Steam Reviews](https://www.kaggle.com/luthfim/steam-reviews-dataset)
Steam is a gaming platform that acts as a third-party medium to sell games online and download them. Users are generally aged between 18 to 30. Consequently, the culture around Steam is built on sarcasm, memes, and wit. This community-driven content often informs other users’ purchases and is also monitored by developers and publishers in order to glean opinions on specific aspects of the game which can be patched or improved in updates to the game. Steam Community allows users to post reviews of games once they have played them.Instead using a 5-star rating system, players are asked to provide their feeling about the game as Recommended , or Not Recommended. The number of playing hours of the reviewed game, the number of games played, and the number of previously posted reviews by the reviewer at this moment are shown aside from reviews. The positive review rate is displayed on the Steam Store page of the game, to advise potential customers.  

Steam Review Dataset is a binary sentiment classification dataset that extracts contents from the steam website containing over hundreds of millions reviews in multiple languages labeled by Steam community members. In our paper, we downloaded steam reviews from kaggle. The dataset contains eight features: date_posted, funny, helpful, hour_played, is_early access review, recommendation, review, title. 
 
## Background
### Research in recent years using steam dataset:

Steamvox(2019) is a platform built to scrape using steam reviews, clean data using NLTK, SpaCy, gensim, Syntok,do topic modelling using Latent Dirichlet Allocation(LDA) and analyze reviews using VADER SentimentIntensityAnalyser from Steam to identify topics and its sentiment for each topic. It is currently focusing on the Total war game: Three kingdoms. Besides it capacity that only be able to focus on analysing one game for the time being, it also observed other problems like: spam reviews are common, often hold no meaning and usually be posted as “spam bomb game” among reviewers; most reviews are too short to do further analysis; vedar analyzer usually failed on discern sarcasms; some reviewers do not use punctuation, and this can affect tokenization process.

The paper *A Study on Video Game Review Summarization* (2019) examines aspect-based summarization and sentiment analysis applied on the game reviews and also offers an evaluation process on the performance of the summarization task aiming to minimize supervision that constantly performs in the previous research. The preprocessing includes converting reviews into tf-idf vectors, tokenization, stopword removal and lemmatization. The experiment set up most frequent words in 5 clusters and applied k-means clustering to perform aspect extraction and Aspect labelling.  Then, they use VADER analyzer to  combine lexical features to sentiment scores with a set of five heuristics. It did not fully resolve the pitfalls stated previously.

In *Summarizing Game Reviews: First Contact* (2020), its summarization pipeline includes : Pre-processing and parsing, Topic Modeling,  Sentiment Analysis, summarization. The first two are based on keyword detection and clustering, the second step performs sentimental analysis and the third step is bi-directional BERT model. The algorithm then uses the generated embedding to train  binary Ridge Logistic Regression classifiers in each aspect.  Each candidate sentence gains a confidence score. Sentences with a high prediction confidence score will be selected  as summarization candidates. This model also displays aspect extraction and Aspect labelling as the previous paper, but it includes sentiment analysis as a feature for summarization. The limitation of the pipelines lies in the mixture of sentiments from users about various features, and it makes labelling and extraction processes harder to get a clear result. 

In *Recommender System: Rating predictions of Steam Games Based on Genre and Topic Modelling* (2020), the research focuses on implementing a genre-based and topic modeling recommender system to predict rating of games. The process includes, data cleaning, convert playtime data to rating, topic extraction by LDA (as previous paper), implementing K-NN algorithm to determine the game rating target, computation of user and item similarity, using RMSE for model evaluation. However, the evaluation shows non-outstanding performance using genre and topic modelling for games recommender systems. 

From these papers, we can observe some similarities in the project design. For example, VADAR analyzer is used in sentiment analysis, LDA used to do Topic Modelling, performing clustering according to each setup clustering tables. (e.g. most frequent words clusters table) Sentiment analysis, topic modeling or genre classification are used as features while implementing text summarization, sentiment analysis, or recommender.

In our project, we scrutinized the review texts and decided to focus on text analysis because we noticed that these papers did not focus on expanding contractions, replacing video game slang. So it will be our focus to make some breakthrough from the previous research progress, and we will also implement ensemble learning for our project design.
 
## Our project design
### Detail design of features and analysis:

There will be a final website where a user submits a link to an IGN video game review: From here, we web scrape the review. There will be three different models used to
  1. Summarize the review 
  2. Classify the video game as "recommended" or "not recommended"
  3. Perform speech synthesis

Our output will derive from different datasets for the same subject to show different perspectives.
For increment 1,our project is based on Sentiment Classification on Steam Reviews regarding the steam reviews dataset (text classification) - (Our increment is working on part B first). The paper uses a couple of different models (Linear SVM, Log Reg, Naive Bayes) and explores features in the steam dataset in addition to the reviews text (such as hours played, number of people who found the review helpful). We scrutinized research limitations and limitations presented in the previous papers investigating the same dataset, and found the review text analysis itself is the core problem that causes previous limitations, therefore we decided to focus on the text analysis to improve the outcome. Besides, considering the output consistency for our website, focusing on analyzing the review text could be the best strategy for the current stage. As a result, we weren't able to do a direct model comparison with the paper; however, we did use the same models and checked performance when processing video game slang for this was one of the limitations. The paper did not consider video game slang, and we think it is an important part that reviews could potentially be misinterpreted. Incorporate with relatively comprehensive machine learning models design, we hope we can provide some innovations to the overall video game review projects improvement.

### Implementation:

We build six different models and perform cross validation for each model with hyper-parameter tuning. After choosing our best model, we will do a final evaluation using our test data.
Here are the models we will be working with:
1.	Naive Bayes
2.	Linear SVM
3.	Logistic Regression
4.	Random Forest
5.	Ensemble Learning
1.	Linear SVM, Logistic Regression, Random Forest - Hard Voting
2.	Naive Bayes, Logistic Regression, Random Forest - Soft Voting

### Preliminary Result:
After looking at our F1 scores, Accuracy, Average, and Weighted Average, it's clear that our first ensemble came out on top ever so slightly. As such, we will use it as our final model and evaluate it on our final test set. Our final model performed well overall, really close to our validation score! Although the F1-score was slightly low for our Not Recommended class, the model was able to perform up to par compared to our training scores. We will be using this model for the "Text Classification" Component of our final project.

### Project Management:
Implementation status report:
- Work completed:
  - Part 2 - video game review text classification 
  - Responsibilities
    - EDA / Text Processing - YingHsuan Lo
    - Model training / Demo for Incrememnt 1 - Daniel
    - Documentation - YingHsuan Lo, Daniel Mata
  - Contributions - 50% - 50%

- Work to be completed:
  - Part 1 - video game summarization 
  - Part 3 - Speech synthesis
  - Front-end website
  - Responsibility (Next focus is Part 1, Text Summarization)
    - Website (Front End) - YinhHsuan Lo 
    - EDA / Text Processing - YingHsuan Lo
    - Build Model - Daniel
    - Documentation - YingHsuan Lo, Daniel Mata
  - Contributions - 50% - 50%
  - Issues / Concerns - We are going to discuss the integration of the project and the details of the potential adjustment according to requirements and project implementation.

Dataset 1: [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset)
This dataset contains articles taken from WikiHow paired with summaries of said articles. This will be used to train our model for summarizing text. The object here will be to perform extractive summarization.
 
### Text-to-Speech Component

Dataset 3 : [CSS10](https://paperswithcode.com/dataset/css10)

This dataset is a collection of Single Speaker Speech Datasets for 10 Languages. It is composed of short audio clips from LibriVox audiobooks and their aligned texts.

Ultimately, we wish to create a website that can integrate all three tasks such that when a user inputs the link of a certain video game review (from IGN.com), an output from all three tasks is performed. From this, we would like to additionally perform web scraping; the process would like so:

→ Web Scrape text
→ Perform text summarization
→ Classify as recommended or not based on summarization
→ Perform Text-to-Speech on previous two outputs
→ Display on website

## Features

Dataset 1:
1.	Articles of how to perform certain task
2.	Content summary
Dataset 2:
1.	User reviews of a given video game. Note: these reviews vary in length
2.	Binary classification: Recommended vs. Not Recommended.
3.	TF-IDF

Dataset 3:
1.	Audio clips read by a single speaker extracted from LibriVox for ten languages.
2.	Text-based script on audio clips.


References
-	[Resource for Text Classification](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/)
-	[Resource for Text Summarization](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f)
-	[Paper describing WikiHow dataset](https://arxiv.org/pdf/1810.09305.pdf)
-	[Speech synthesis resource / CSS10 / Dataset paper](https://paperswithcode.com/task/speech-synthesis)
-	[Resource guide for speech synthesis model](https://heartbeat.comet.ml/a-2019-guide-to-speech-synthesis-with-deep-learning-630afcafb9dd?gi=cc07293e61ba)
-	[IGN Review website for user integration](https://www.ign.com/reviews/games)
-	[Sentiment Analysis of Steam Review Datasets using Naive Bayes and Decision Tree Classifier (2018)](https://core.ac.uk/download/pdf/159108993.pdf)
-	[Steam Review Dataset - new, large scale sentiment dataset (2016)](https://www.researchgate.net/publication/311677831_Steam_Review_Dataset_-_new_large_scale_sentiment_dataset)
-	[Steamvox: SteamVox is built to obtain the “Voice of the Player” from players’ (2019)](https://github.com/alfredtangsw/steamvox)
-	[A Study on Video Game Review Summarization (2019)](https://aclanthology.org/W19-8906.pdf)
-	[Summarizing Game Reviews: First Contact (2020)](http://ceur-ws.org/Vol-2844/games5.pdf)
-	[An Empirical Study of Game Reviews on the Steam Platform (2019)](https://seal-queensu.github.io/publications/pdf/EMSE-Dayi-2019.pdf)
-	[Recommender System: Rating predictions of Steam Games Based on Genre and Topic Modelling (2020)](https://ieeexplore.ieee.org/document/9140194)

