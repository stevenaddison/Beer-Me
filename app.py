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

st.sidebar.write( '<h1 style="text-align:center">Beer Recommender</h1>' , unsafe_allow_html=True)

st.sidebar.caption('by [Steven Addison](https://www.linkedin.com/in/addisonse/)')

st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">How It Works</h1>' , unsafe_allow_html=True)
  
st.sidebar.write('''
Recommendations are generated from a list of 82000 unique beers from 5400 different breweries across the United States using 
Natural Language Processing to calculate similarity based on individual reviews by users on 
[Beer Advocate](https://www.beeradvocate.com).
''')

st.sidebar.image(star_image)

st.sidebar.write( '<h1 style="text-align:center">Curious About This Project?</h1>' , unsafe_allow_html=True)

st.sidebar.write(
'''Find my notebook outlining my work on my [github](https://github.com/stevenaddison/Capstone).
''')


####### Modeling #######

# loading the data
cont_df = pd.read_csv('data/cont_df.csv', index_col='beer_id')

# null value sanity check
cont_df = cont_df.dropna(subset=['clean_text'])

# defining the dataframe where we will source our results
result_df = cont_df[['beer_name','style','broad_style','brewery_name','city','state']]

#making sure the output is pretty
result_df = result_df.rename(columns={'beer_name':'Name','style':'Style',
                                      'broad_style':'Broad Style',
                                      'brewery_name':'Brewery','city':'City',
                                      'state':'State'})

# defining the dataframe we will use to model
model_df = cont_df[['abv','score','clean_text','broad_style']]

style_input = st.selectbox("What style beer are you looking for?", (result_df['Broad Style'].unique()))

beer_string = "Which " + style_input + " have you enjoyed recently?"

beer_input = st.selectbox(beer_string, sorted(result_df[result_df['Broad Style'] == style_input]['Name'].unique()))

n_recs = st.number_input("How many recommendations would you like?", max_value=10)
 
# cosine similarity model
def cos_sim(style_input,beer_input, n_recs):

    style_df = model_df[model_df['broad_style'] == style_input]
        
    tf = TfidfVectorizer(max_features=500, ngram_range=(1,3))
    dtm = tf.fit_transform(style_df['clean_text'])
    dtm = pd.DataFrame(dtm.todense(), columns=tf.get_feature_names(), index = style_df.index)
    style_df = style_df.merge(dtm, left_index=True, right_index=True)
    style_df = style_df.drop(columns=['broad_style','clean_text'])
    col_names = ['abv_x', 'score']
    features = style_df[col_names]
    features = MinMaxScaler().fit_transform(features.values)
    style_df[col_names] = features
            
    beerix = cont_df.loc[cont_df['beer_name'] == beer_input].index.values
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

    st.write(display_results)