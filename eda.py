def eda(df):
    """Function to perform some basic EDA on my datasets"""
    
    #Inspect the first 5 rows
    display(df.head())
    print("\n")
    
    # Count of non-null values, datatypes, and total entries
    display(df.info())
    print("\n")
    
    # Check descriptive statistics
    display(df.describe())
    print("\n")
    
    # Check value counts
    for c in df.columns:
        print ("---- %s ----" % c)
        print (df[c].value_counts())
        print("\n")
    
    # Print null values
    display(df.isna().sum())
    print('Total Null Count:', df.isna().sum().sum())
    
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
