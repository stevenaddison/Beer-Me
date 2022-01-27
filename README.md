![header](https://github.com/stevenaddison/Capstone/blob/main/images/beerflight.jpg)
# Capstone: Beer Recommender System
Author: Steven Addison


## Overview
This project analyzes beer review data of 60,000 unique beers and recommends the top similar styles analyzing Review Text with a tf-idf vectorizer, Style, ABV, and Overall Rating.

## Business Problem
As someone who has spent many years in the restaurant industry I often dread hearing “what do you recommend”, it always has felt like a loaded question to me seeing how one never has enough time to learn about a customer’s likes and taste in order to give accurate guidance. That is why a content-based recommender system such as this one can take all the guessing work out of the equation so that a customer can enjoy a beverage that has been selected for them based on similar items other enthusiasts have previously reviewed and enjoyed. 

## Data Understanding

For this project I used three datasets sourced from [Kaggle](https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews) that contains information gathered from [Beer Advocate](https://www.beeradvocate.com/). For reproducibility one will have to download the data from the source, as it is too large to host on Github. 

## Data Preparation
Starting with over 9 million unique entires in the `reviews` database, I was able to generate recommendations for over sixty thousand unique beers from fifty four hundred different breweries across the United States.  Keeping only data relvant `text` data, where `country` is United States, and not currently `retired`, I group  the reviews by `beer_id`, keeping all `text` data, the mean `score`, `abv`, and `style`. Then using the text preprocessing package [Texthero](https://texthero.org/) I clean the reviews, and 
create a document-term matrix using `TfidfVectorizer`.

![stylecounts](https://user-images.githubusercontent.com/92377177/151299722-f2d3b890-6858-4273-a3e5-6cb2da9a825f.png)

## Modeling & Visualizations
Using the document-term matrix and one hot encoding, I built content-based recommender systems using `Cosine Similarity`, `Linear Kernel`, and `K-Nearest Neighbors`.

Below, wordclouds are generated on my modeling beer and its top recommendation.
![superfuzzcancloud](https://user-images.githubusercontent.com/92377177/151300073-0d9fc734-e747-4997-bb22-982061713d23.png)
![citrusinesiscancloud](https://user-images.githubusercontent.com/92377177/151300080-38418471-2e01-4daf-a276-59b2fb89a8a9.png)


## Deployment
From the functions created in my notebook I built an app using `Streamlit`, deployed on [Heroku](https://beer-me-recommender.herokuapp.com/). 

https://user-images.githubusercontent.com/92377177/151299653-089d0c87-02c8-4216-938b-68c508726d5c.mp4

First you select what style beer you would like, these are broader categories than using every beer’s exact style. If you don’t see the style you are looking for an expandable dictionary is provided so you may see what broad category it may fall under.

Next, type in the beer you would like  your recommendation based off of. Every beer is listed with their respective brewery as well so you may search for styles more easily. 

Lastly, select how many recommendations you are looking for, and then hit the `beer me` button! Once these inputs are selected the app will slice the data based on the selected style and perform the tf-idf vectorization on the cleaned review text, scale the numerical features, and return your results, as simple as that. 

Unfortunately the app takes too much memory to currently be ran with Heroku's limited conditions, however you can run it locally if cloned from my github repo.  

## Running The App Locally

Clone this git repo, then install the specified requirements with pip.

```
git clone https://github.com/stevenaddison/Capstone
cd Capstone
pip install -r requirements.txt
```

Run the app.

```
streamlit run app.py
```

## Conclusions & Next Steps
As extensive as this may seem, there is always more data that can be gathered. Ideally I would like to get more current data for as many beers as possible to keep the model as up to date as possible. I would also be interested in gathering more features to use such as IBU, and the varieties of hops and malts used to brew each beer. In addition to that, I would like to create more specific data slicing options in the app such as being able to get recommendations based on state or even a brewery level. Lastly I would want to implement this model for other consumables such as whole bean coffee or wine, if you can reviewed it ,I can use it.


## Repository Structure
```
├── [data]
├── [images]
├── .gitignore
├── README.md
├── finalnotebook.ipynb
├── finalnotebook.pdf
└── presentation.pdf
```
