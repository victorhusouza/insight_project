import pandas as pd
import streamlit as st
import numpy as np
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

pd.set_option('display.float_format', lambda x : '%.2f' % x)

#page layout
st.set_page_config( layout='wide' )

st.title('House Rocket Case')
st.markdown('House rocket is a fictitious company which its core business is to buy real state with lower price, renovate then sell it to a higher price. '
            'Its CEO wants to maximize the his profit,\n in order to achieve that he hired a data scientist, '
            'to analyze a dataset and look for the best opportunities with the higher'
            'probability of profit')

@st.cache(allow_output_mutation =  True)
def get_data(path):

    data = pd.read_csv(path)

    return data

def clean_transform(df):

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['sqft_lot15', 'sqft_living15'])

    return df

def new_features(df):

    # feature for hypothesis 1
    df['waterview'] = df['waterfront'].apply(lambda x: 'no' if (x == 0) else 'yes')

    # feature for hypothesis 2
    df['yr_b_mean'] = df['yr_built'].apply(lambda x: '< 1955' if (x < 1955) else '> 1955')

    # feature for hypothesis 3
    df['has_basement'] = df['sqft_basement'].apply(lambda x: 'no' if (x == 0) else 'yes')

    # feature for hypothesis 4
    df['year'] = pd.to_datetime(df['date']).dt.year

    # feature for hypothesis 5
    df['month'] = pd.to_datetime(df['date']).dt.month

    # add feature
    df['price_m2'] = df['price'] / (df['sqft_lot'] / 10.764)

    return df

def data_overview(df):
    # Average metrics

    c1, c2 = st.columns((2, 2))

    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df[['zipcode', 'price']].groupby('zipcode').mean().reset_index()
    df3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    data = pd.merge(m2, df4, on='zipcode', how='inner')

    data.columns = ['zipcode', 'total house', 'price', 'sqrft living', 'Price/m2']

    c1.title('Average metrics')
    c1.dataframe(data, width=700, height=700)


    # select only data types equals to int64 and float64
    num_attri = df.select_dtypes(include=['int64', 'float64'])

    # exclude the id column
    num_attri = num_attri.iloc[:, 1:]

    mean = pd.DataFrame(num_attri.apply(np.mean))
    median = pd.DataFrame(num_attri.apply(np.median))
    std = pd.DataFrame(num_attri.apply(np.std))
    max_ = pd.DataFrame(num_attri.apply(np.max))
    min_ = pd.DataFrame(num_attri.apply(np.min))

    # concatenate all created variables
    df0 = pd.concat([max_, min_, mean, median, std], axis=1).reset_index()

    # rename columns
    df0.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    c2.title('Descriptive Analysis')
    c2.dataframe(df0, height=800)

    return None

def bq1(df):

    # Business Question 1
    st.subheader('1. Which is the real state that House Rocket should buy and at what price?')

    # create feature for price median and concatenate with dataset
    data = pd.DataFrame()

    data = df[['zipcode', 'price']].groupby('zipcode').median().reset_index()

    df2 = pd.merge(df, data, on='zipcode', how='inner')

    df2 = df2.rename(columns={'price_x': 'price', 'price_y': 'price_median'})

    # set conditions to buy or not
    for i in range(len(df2)):
        if (df2.loc[i, 'price'] < df2.loc[i, 'price_median']) & (df2.loc[i, 'condition'] >= 3):
            df2.loc[i, 'status'] = 'buy'
        else:
            df2.loc[i, 'status'] = 'do not buy'



    # filters

    st.sidebar.header('Business Question 1 Filters')

    # filter for zipcode
    f_zipcode = st.sidebar.multiselect('Enter zipcode', df2['zipcode'].unique())

    if (f_zipcode != []) :

        df2 = df2.loc[df2['zipcode'].isin(f_zipcode),:]

    else:
        df2 = df2.copy()

    # filter by price range
    min_price = int(df2['price'].min())
    max_price = int(df2['price'].max())
    avg_price = int(df2['price'].mean())

    # data filtering
    f_price = st.sidebar.slider('Maximun Price', min_price, max_price, avg_price)
    df2 = df2[df2['price'] < f_price]

    # filter by price median range
    min_price_median = int(df2['price_median'].min())
    max_price_median = int(df2['price_median'].max())
    avg_price_median = int(df2['price_median'].mean())

    # data filtering
    f_price_median = st.sidebar.slider('Price Median Maximun', min_price_median, max_price_median, avg_price_median)
    df2 = df2[df2['price_median'] < f_price_median]

    # filter for condition
    f_condition = st.sidebar.multiselect('Enter Condition', sorted(set(df['condition'].unique())))
    if (f_condition != []) :

        df2 = df2.loc[df2['condition'].isin(f_condition),:]

    else:
        df2 = df2.copy()

    # filter only houses to buy
    df2 = df2[df2['status'] == 'buy']

    st.dataframe(df2[['id', 'date', 'zipcode', 'condition', 'price', 'price_median', 'status']].sort_values(by = 'price', ascending = False))

    return None

def bq2(df):

    # Business Question 2

    # create feature for price median and concatenate with dataset
    data = pd.DataFrame()

    data = df[['zipcode', 'price']].groupby('zipcode').median().reset_index()

    df2 = pd.merge(df, data, on='zipcode', how='inner')

    df2 = df2.rename(columns={'price_x': 'price', 'price_y': 'price_median'})

    # set conditions to buy or not
    for i in range(len(df2)):
        if (df2.loc[i, 'price'] < df2.loc[i, 'price_median']) & (df2.loc[i, 'condition'] >= 3):
            df2.loc[i, 'status'] = 'buy'
        else:
            df2.loc[i, 'status'] = 'do not buy'

    st.subheader('2. Once bought the real state, when is the best moment to sell and at what price?')

    # categorize by season available
    df2['season'] = df2['month'].apply(lambda x: 'Spring' if (x >= 3) & (x <= 5) else 'Summer' if (x >= 6) & (x <= 8)
    else 'Autumn' if (x >= 9) & (x <= 11) else 'Winter')

    df3 = df2.copy()
    b4 = df3[df3['status'] == 'buy']

    b2 = b4[['season', 'zipcode', 'price']].groupby(['zipcode', 'season']).median().reset_index()

    b5 = pd.merge(b4, b2, on=['season', 'zipcode'], how='inner')

    b5 = b5.rename(columns={'price_x': 'price', 'price_y': 'price_median_season'})

    for i in range(len(b5)):
        if (b5.loc[i, 'price_median_season'] < b5.loc[i, 'price']):

            b5.loc[i, 'Sell Price'] = (b5.loc[i, 'price'] * 1.3)

            b5.loc[i, 'Profit'] = (b5.loc[i, 'price'] * 0.3)

        else:

            b5.loc[i, 'Sell Price'] = (b5.loc[i, 'price'] * 1.1)

            b5.loc[i, 'Profit'] = (b5.loc[i, 'price'] * 0.1)



    # filters
    st.sidebar.header('Business Question 2 Filters')

    # filter by season
    f_season = st.sidebar.multiselect('Enter season', b5['season'].unique())

    if (f_season != []):

        b5 = b5.loc[b5['season'].isin(f_season), :]

    else:
        b5 = b5.copy()

    # filter by price range
    min_price = int(b5['price'].min())
    max_price = int(b5['price'].max())
    avg_price = int(b5['price'].mean())

    f_price = st.sidebar.slider('Enter Maximun Price ', min_price, max_price, avg_price)
    b5 = b5[b5['price'] < f_price]

    # filter by profit
    min_profit = int(b5['Profit'].min())
    max_profit = int(b5['Profit'].max())
    avg_profit = int(b5['Profit'].mean())

    f_profit = st.sidebar.slider('Enter Maximun Price ', min_profit, max_profit, avg_profit)
    b5 = b5[b5['Profit'] < f_profit]


    st.dataframe(b5[['id', 'date','season', 'zipcode', 'price', 'price_median', 'price_median_season', 'Sell Price', 'Profit']])

    return None

if __name__ == '__main__':
    #ETL
    # extraction

    # get data
    path = 'kc_house_data.csv'

    # get file
    data = get_data(path)

    # transformation

    data = clean_transform(data)

    data = new_features(data)

    st.write(data.columns)

    data_overview(data)

    st.header('Business Questions')

    bq1(data)

    bq2(data)