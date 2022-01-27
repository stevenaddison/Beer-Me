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
Starting with over 9 million unique entires in the `reviews` database, I was able to generate recommendations for over sixty thousand unique beers from fifty four hundred different breweries across the United States.  Keeping only data relvant `text` data, where `country` is United States, and not currently `retired`, I group  the reviews by `beer_id`, keeping all `text` data, the mean `score`, `ABV`, and `style`. Then using the text preprocessing package [Texthero](https://texthero.org/) I clean the reviews, create a document-term matrix using `TfidfVectorizer`.

## Modeling & Visualizations

## Conclusions & Recommendations

## Next Steps

## <a id="Sources">Sources</a>

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
