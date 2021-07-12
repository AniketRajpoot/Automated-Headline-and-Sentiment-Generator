# Automated-Headline-and-Sentiment-Generator
# Overview
**Digital content** is expanding at a very rapid pace. Many activities that experts undertake today involve the ability to process digital content and synthesize them to make decisions.


We were provided with a dataset consisting of news articles and tweets which belong to either **mobile technology**. The problem statement consisted of 3 parts : 

1. Develop an intelligent system that could first identify the theme of tweets and articles.
2. If the theme is **mobile technology** then it should identify the sentiments against a brand (at a tweet/paragraph level).
3. Finally a one-sentence headline of max of 20 words for articles that follow the mobile technology theme was needed to be generated. A headline for tweets is not required.

We approached each of these sub problems separately, using three different models for each task. We used state of the art **Transformers** for the purpose of
classification of tweets/articles and generation of headlines. We used **VADER** for Sentiment Extraction which is a sentiment analysis model. The specific
approaches are discussed in more detail in later sections.
