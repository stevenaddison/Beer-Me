import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


from PIL import Image

####### Header #######
header_image = Image.open('images/title.PNG')
star_image = Image.open('images/star.PNG')
st.image(header_image)

####### Sidebar #######
styledict= {
'German Doppelbock':'Bock',
'German Maibock': 'Bock',
'German Bock': 'Bock',
'German Weizenbock': 'Bock',
'German Eisbock': 'Bock', 
'German Altbier':'Brown Ale',
'American Brown Ale':'Brown Ale',
'English Brown Ale':'Brown Ale',
'English Dark Mild Ale':'Brown Ale', 
'Belgian Dubbel':'Dark Ale',
'German Roggenbier':'Dark Ale',
'Scottish Ale':'Dark Ale',
'Winter Warmer':'Dark Ale', 
'American Amber / Red':'Dark Lager',
'American Amber / Red Ale':'Dark Lager',
'European Dark Lager':'Dark Lager',
'German Märzen / Oktoberfest':'Dark Lager',
'Munich Dunkel Lager':'Dark Lager',
'German Rauchbier':'Dark Lager',
'German Schwarzbier':'Dark Lager',
'Vienna Lager':'Dark Lager', 
'American IPA':'India Pale Ale',
'Belgian IPA':'India Pale Ale',
'American Brut IPA':'India Pale Ale',
'English India Pale Ale (IPA)':'India Pale Ale',
'American Imperial IPA':'India Pale Ale',
'New England IPA':'India Pale Ale',
'American Black Ale':'India Pale Ale',
'English Bitter':'Pale Ale',
'English Extra Special / Strong Bitter (ESB)':'Pale Ale',
'Belgian Blonde Ale':'Pale Ale',
'American Blonde Ale':'Pale Ale',
'French Bière de Garde':'Pale Ale',
'Belgian Saison':'Pale Ale',
'German Kölsch':'Pale Ale',
'English Pale Ale':'Pale Ale',
'English Pale Mild Ale':'Pale Ale',           
'American Pale Ale (APA)':'Pale Ale',
'Belgian Pale Ale':'Pale Ale',
'American Amber / Red Lager':'Pale Ale',
'Irish Red Ale':'Pale Ale',
'American Adjunct Lager':'Pale Lager',
'American Light Lager ':'Pale Lager',
'European Export / Dortmunder':'Pale Lager',
'European Pale Lager':'Pale Lager',
'European Strong Lager':'Pale Lager',
'German Helles':'Pale Lager',
'German Kellerbier / Zwickelbier':'Pale Lager',
'American Light Lager':'Pale Lager',
'American Malt Liquor':'Pale Lager',
'Bohemian Pilsener':'Pale Lager',
'German Pilsner':'Pale Lager',
'American Imperial Pilsner':'Pale Lager',
'American Lager':'Pale Lager',
'American Porter':'Porter',
'English Porter':'Porter',
'Baltic Porter':'Porter',
'American Imperial Porter':'Porter',
'Smoke Porter':'Porter',
'Robust Porter':'Porter',
'American Imperial Stout':'Stout',
'American Stout':'Stout',
'English Sweet / Milk Stout':'Stout',
'Russian Imperial Stout':'Stout',
'English Oatmeal Stout':'Stout',
'Irish Dry Stout':'Stout',
'English Stout':'Stout',
'Foreign / Export Stout':'Stout',
'American Barleywine':'Strong Ale',
'British Barleywine':'Strong Ale',
'English Old Ale':'Strong Ale',
'Belgian Quadrupel (Quad)':'Strong Ale',
'American Imperial Red Ale':'Strong Ale',
'Scotch Ale / Wee Heavy':'Strong Ale',
'American Strong Ale':'Strong Ale',
'Belgian Dark Ale':'Strong Ale',
'Belgian Strong Dark Ale':'Strong Ale',
'Belgian Strong Pale Ale':'Strong Ale',
'English Strong Ale':'Strong Ale',
'Belgian Tripel':'Strong Ale',
'American Wheatwine Ale':'Strong Ale',
'American Dark Wheat Ale':'Wheat Beer',
'American Pale Wheat Ale':'Wheat Beer',
'German Dunkelweizen':'Wheat Beer',
'German Kristalweizen':'Wheat Beer',
'German Hefeweizen':'Wheat Beer',
'Belgian Witbier':'Wheat Beer',
'American Brett':'Wild/Sour Beer',
'Belgian Faro':'Wild/Sour Beer',
'Belgian Fruit Lambic':'Wild/Sour Beer',
'Belgian Gueuze':'Wild/Sour Beer',
'Belgian Lambic':'Wild/Sour Beer',
'Berliner Weisse':'Wild/Sour Beer',
'Flanders Oud Bruin':'Wild/Sour Beer',
'Flanders Red Ale':'Wild/Sour Beer',
'Leipzig Gose':'Wild/Sour Beer',
'American Wild Ale':'Wild/Sour Beer',
'Wild/Sour Beers':'Wild/Sour Beer',
'Bière de Champagne / Bière Brut':'Hybrid Beer',
'Braggot':'Hybrid Beer',
'California Common / Steam Beer':'Hybrid Beer',
'American Cream Ale':'Hybrid Beer',
'Chile Beer':'Specialty Beer',
'Farmhouse Ale - Sahti':'Specialty Beer',
'Fruit and Field Beer':'Specialty Beer', 
'Happoshu':'Specialty Beer',
'Herb and Spice Beer':'Specialty Beer',
'Russian Kvass':'Specialty Beer',
'Japanese Rice Lager':'Specialty Beer',
'Low Alcohol Beer':'Specialty Beer',
'Pumpkin Beer':'Specialty Beer',
'Rye Beer':'Specialty Beer', 
'Smoke Beer':'Specialty Beer',
'Scottish Gruit / Ancient Herbed Ale':'Specialty Beer',
'Finnish Sahti': 'Specialty Beer'
}

dictdf = pd.DataFrame.from_dict(styledict, orient='index')

st.sidebar.write( '<h1 style="text-align:center">Beer Me!</h1>' , unsafe_allow_html=True)
st.sidebar.write( '<h2 style="text-align:center">a content-based recommender system</h2>' , unsafe_allow_html=True)
st.sidebar.caption('by [Steven Addison](https://www.linkedin.com/in/addisonse/)')

st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">How It Works</h1>' , unsafe_allow_html=True)
  
st.sidebar.write('''
Recommendations are generated from a dataset containing 60,000 unique beers from 5,100 different breweries across the United States using 
Natural Language Processing to calculate similarity based on [Beer Advocate](https://www.beeradvocate.com) Reviews, ABV, Style, and Review Score.
''')


st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">Curious About This Project?</h1>' , unsafe_allow_html=True)

st.sidebar.write(
'''Find my notebook outlining my work on my [github](https://github.com/stevenaddison/Capstone).
''')

####### Modeling #######
result_df = pd.read_csv('data/result_df.csv', index_col = 'beer_id')

style_input = st.selectbox("What style beer are you looking for?", sorted((result_df['broad_style'].unique())))

with st.beta_expander('Click here if your desired style is not showing to see what category it falls into!', expanded=False):
    st.write(dictdf)

beer_string = "Which " + style_input + " have you enjoyed recently?"

beer_input = st.selectbox(beer_string, sorted(result_df[result_df['broad_style'] == style_input]['brewplusbeer'].unique()))

n_recs = st.number_input("How many recommendations would you like?", max_value=10)

def cos_sim(style_input,beer_input, n_recs):
    if style_input == 'India Pale Ale':
            style_df = pd.read_csv('data/IndiaPaleAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Pale Ale':
            style_df = pd.read_csv('data/PaleAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Stout':
            style_df = pd.read_csv('data/Stout.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Wild/Sour Beer':
            style_df = pd.read_csv('data/WildSout.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Strong Ale':
            style_df = pd.read_csv('data/StrongAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Dark Lager':
            style_df = pd.read_csv('data/DarkLager.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Pale Lager':
            style_df = pd.read_csv('data/PaleLager.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Wheat Beer':
            style_df = pd.read_csv('data/WheatBeer.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Porter':
            style_df = pd.read_csv('data/Porter.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Specialty Beer':
            style_df = pd.read_csv('data/SpecialtyAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Brown Ale':
            style_df = pd.read_csv('data/BrownAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Dark Ale':
            style_df = pd.read_csv('data/DarkAle.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Bock':
            style_df = pd.read_csv('data/Bock.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 
    if style_input == 'Hybrid Beer':
            style_df = pd.read_csv('data/HybridBeer.csv', index_col= 'beer_id')
            style_df.drop(columns= 'Unnamed: 0', inplace=True) 

    beerix = result_df.loc[result_df['brewplusbeer'] == beer_input].index.values
    y = np.array(style_df.loc[beerix[0]])
    y = y.reshape(1, -1)

    cos_sim = cosine_similarity(style_df, y)
    cos_sim = pd.DataFrame(data=cos_sim, index=style_df.index)
    results = cos_sim.sort_values(by = 0, ascending=False)
    nresultsid = results.head(n_recs+1).index.values[1:]
    nresults_df = result_df.loc[nresultsid]
    nresults_df = nresults_df.style.hide_index()

    return nresults_df

display_recommendation_now = st.button('Beer me!')
if display_recommendation_now:
    display_results = cos_sim(style_input,beer_input, n_recs)

    # Display a static table
    st.table(display_results)