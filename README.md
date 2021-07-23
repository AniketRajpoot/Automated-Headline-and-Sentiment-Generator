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

## Models Used 
- [:heavy_check_mark:] [Text Classification : **XLM-RoBERTa**](Text_classification_code/)
- [:heavy_check_mark:] [Sentiment Extraction : **VADER**](Brand_and_sentiment_identification_code/)
- [:heavy_check_mark:] [Headline Generation : **MT5**](Headline_generation_code/)

## Features provided 
- [x] Implementation of script and complete procedure for Text Classification with instructions. 
- [ ] Implementation of scripts for headline generation and sentiment extraction. 
- [ ] Implementation of complete single pipeline performing all the three steps in order. 

# Setup 

Run the following commands to set-up environment:  
```
git clone https://github.com/AniketRajpoot/Automated-Headline-and-Sentiment-Generator.git  
cd Automated-Headline-and-Sentiment-Generator  
pip install -r requirements.txt  
```
## Pre-trained checkpoints:  

### **XLM-RoBERTa**

Run the following command:
```
gdown --id 1mBhGHYOTnikOJD3KOBK1s_FuCjaiUR1a
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1mBhGHYOTnikOJD3KOBK1s_FuCjaiUR1a/view?usp=sharing
```

### **MT5**

Run the following command:
```
!gdown --id 1ncA3AMBPEFvfv8xMdLrl6vCSzNUu3sw9
```
Alternatively, the link to the same is given below:
```
https://drive.google.com/file/d/1ncA3AMBPEFvfv8xMdLrl6vCSzNUu3sw9/view?usp=sharing
```

## Scripts

### Text Classification

```
python -u scripts/predict_class.py --file <FILEPATH> <OR>  
python -u scripts/predict_class.py --sen <SENTENCE>  
```
Sample run:
```
python -u scripts/predict_class.py --file 'sample_article.txt' 
```

### Headline Generation

```
python -u scripts/predict_headline.py --file <FILEPATH> <OR>  
python -u scripts/predict_headline.py --sen <SENTENCE>  --num_sentences <NO OF HEADLINES>  
```
Sample run:
```
python -u scripts/predict_headline.py --file 'sample_article_2.txt' --num_sentences 5 
```

# Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one.
