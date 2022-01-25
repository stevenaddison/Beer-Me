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
styledict= {'German Doppelbock':'Bock',
'German Maibock': 'Bock',
'German Bock': 'Bock',
'German Weizenbock': 'Bock',
'German Eisbock': 'Bock', 
'American Black Ale':'Black Ale',
'German Altbier':'Brown Ale',
'American Brown Ale':'Brown Ale',
'English Brown Ale':'Brown Ale',
'English Dark Mild Ale':'Brown Ale', 
'American Cream Ale':'Cream Ale',
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
'English Bitter':'Pale Ale',
'English Pale Mild Ale':'Pale Ale',
'English Extra Special / Strong Bitter (ESB)':'Pale Ale',
'Belgian Blonde Ale':'Pale Ale',
'American Blonde Ale':'Pale Ale',
'French Bière de Garde':'Pale Ale',
'Belgian Saison':'Pale Ale',
'German Kölsch':'Pale Ale',
'English Pale Ale':'Pale Ale',
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
'Wild/Sour Beers':'Wild/Sour Beer'}

dictdf = pd.DataFrame.from_dict(styledict, orient='index')

st.sidebar.write( '<h1 style="text-align:center">Beer Me</h1>' , unsafe_allow_html=True)
st.sidebar.write( '<h2 style="text-align:center">a content-based recommender system</h2>' , unsafe_allow_html=True)
st.sidebar.caption('by [Steven Addison](https://www.linkedin.com/in/addisonse/)')

st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">How It Works</h1>' , unsafe_allow_html=True)
  
st.sidebar.write('''
Recommendations are generated from dataset containing 82000 unique beers from 5400 different breweries across the United States using 
Natural Language Processing to calculate similarity based on individual reviews by users on 
[Beer Advocate](https://www.beeradvocate.com), ABV, Style, and Review Score.
''')


st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">Curious About This Project?</h1>' , unsafe_allow_html=True)

st.sidebar.write(
'''Find my notebook outlining my work on my [github](https://github.com/stevenaddison/Capstone).
''')

####### Modeling #######

# loading the data
cont_df = pd.read_csv('data/cont_df.csv', index_col='beer_id')

#creating brewery and beer column

cont_df['brewplusbeer'] = cont_df['brewery_name'] + " " + cont_df['beer_name']
# null value sanity check
cont_df = cont_df.dropna(subset=['clean_text'])

# defining the dataframe where we will source our results
result_df = cont_df[['beer_name','style','brewery_name','city','state']]

#making sure the output is pretty
result_df = result_df.rename(columns={'beer_name':'Name','style':'Style',
                                      'brewery_name':'Brewery','city':'City',
                                      'state':'State'})

# defining the dataframe we will use to model
model_df = cont_df[['abv','score','clean_text','broad_style']]

style_input = st.selectbox("What style beer are you looking for?", sorted((cont_df['broad_style'].unique())))

with st.beta_expander('Click here if your desired style is not showing to see what category it falls into!', expanded=False):
    st.write(dictdf)

beer_string = "Which " + style_input + " have you enjoyed recently?"

beer_input = st.selectbox(beer_string, sorted(cont_df[cont_df['broad_style'] == style_input]['brewplusbeer'].unique()))

n_recs = st.number_input("How many recommendations would you like?", max_value=10)
 
# cosine similarity model
def cos_sim(style_input,beer_input, n_recs):

    style_df = model_df[model_df['broad_style'] == style_input]
        
    tf = TfidfVectorizer(max_features=300, ngram_range=(1,3))
    dtm = tf.fit_transform(style_df['clean_text'])
    dtm = pd.DataFrame(dtm.todense(), columns=tf.get_feature_names(), index = style_df.index)
    style_df = style_df.merge(dtm, left_index=True, right_index=True)
    style_df = style_df.drop(columns=['broad_style','clean_text'])
    col_names = ['abv_x', 'score']
    features = style_df[col_names]
    features = MinMaxScaler().fit_transform(features.values)
    style_df[col_names] = features
            
    beerix = cont_df.loc[cont_df['brewplusbeer'] == beer_input].index.values
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