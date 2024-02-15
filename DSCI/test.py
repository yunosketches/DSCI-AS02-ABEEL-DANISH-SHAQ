import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 #2
st.set_page_config(layout="wide") #setting page layout

df = pd.read_csv('clean_AS01.csv')

st.title('Olympic Games Dashboard')
st.image('img.jpg', use_column_width=True)

#filettre
by_sport = st.sidebar.multiselect('Select Sport', df['Sport'].unique())
medal = st.sidebar.selectbox('Select Medal', ['Gold', 'Silver', 'Bronze', 'None'])
country = st.sidebar.multiselect('Select Country', df['Team'].unique())
medal_total = st.sidebar.number_input('Minimum Medals Accumulated', min_value=0)
year = st.sidebar.multiselect('Select Year', df['Year'].unique())
season = st.sidebar.multiselect('Select Season', df['Season'].unique())
weight = st.sidebar.slider("Select Weight Range", min_value=df['Weight'].min(), max_value=df['Weight'].max(), value=(df['Weight'].min(), df['Weight'].max()))
height = st.sidebar.slider("Select Height Range", min_value=df['Height'].min(), max_value=df['Height'].max(), value=(df['Height'].min(), df['Height'].max()))
age = st.sidebar.slider("Select Age Range", min_value=df['Age'].min(), max_value=df['Age'].max(), value=(df['Age'].min(), df['Age'].max()))

# set varuable for the filtering
filtered_df = df
if by_sport:
    filtered_df = filtered_df[filtered_df['Sport'].isin(by_sport)]
if medal != 'None':
    filtered_df = filtered_df[filtered_df['Medal'] == medal]
if country:
    filtered_df = filtered_df[filtered_df['Team'].isin(country)]
if year:
    filtered_df = filtered_df[filtered_df['Year'].isin(year)]
if season:
    filtered_df = filtered_df[filtered_df['Season'].isin(season)]
if weight == "Weight":
    filtered_df = filtered_df[(filtered_df['Weight'] >= weight[0]) & (filtered_df['Weight'] <= weight[1])]
elif height == "Height":
    filtered_df = filtered_df[(filtered_df['Height'] >= height[0]) & (filtered_df['Height'] <= height[1])]
elif age == "Age":
    filtered_df = filtered_df[(filtered_df['Age'] >= age[0]) & (filtered_df['Age'] <= age[1])]


#total medal count trial????
total_medal = filtered_df.groupby('Team')['Medal'].count().reset_index()
total_medal.columns = ['Country', 'Total Medals']
total_medal = total_medal[total_medal['Total Medals'] >= medal_total]

#total medal shown
if not total_medal.empty:
    st.subheader('Total Medals Accumulated by Each Country')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Total Medals', y='Country', data=total_medal.sort_values(by='Total Medals', ascending=False))
    st.pyplot()

## start of visualization
col1, col2 = st.columns(2)
with col1:
    st.subheader('1. Distribution of Athletes by Sex')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sex', data=filtered_df)
    st.pyplot()

    st.subheader('3. Age Distribution of Athletes')
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['Age'], bins=20, kde=True)
    st.pyplot()

with col2:
    st.subheader('2. Distribution of Athletes by Medal')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Medal', data=filtered_df, order=['Gold', 'Silver', 'Bronze', 'None'])
    st.pyplot()

    st.subheader('4. Height Distribution of Athletes')
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['Height'], bins=20, kde=True)
    st.pyplot()

col3, col4 = st.columns(2)
with col3:
    st.subheader('5. Weight Distribution of Athletes')
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['Weight'], bins=20, kde=True)
    st.pyplot()

    st.subheader('7. Distribution of Athletes by Season')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Season', data=filtered_df)
    st.pyplot()

with col4:
    st.subheader('6. Number of Athletes per Year')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Year', data=filtered_df)
    plt.xticks(rotation=45)
    st.pyplot()

    st.subheader('8. Top 10 Participating Teams')
    top_teams = filtered_df['Team'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_teams.values, y=top_teams.index)
    st.pyplot()

col5, col6 = st.columns(2)

with col5:
    st.subheader('9. Distribution of Athletes by NOC')
    plt.figure(figsize=(12, 8))
    sns.countplot(y='NOC', data=filtered_df, order=filtered_df['NOC'].value_counts().index[:15])
    st.pyplot()

    st.subheader('11. Distribution of Medals by Year')
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Year', hue='Medal', data=filtered_df)
    plt.xticks(rotation=45)
    st.pyplot()

with col6:
    st.subheader('10. Distribution of Athletes by Sport')
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Sport', data=filtered_df, order=filtered_df['Sport'].value_counts().index[:15])
    st.pyplot()

    st.subheader('12. Distribution of Medals by Sport')
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Sport', hue='Medal', data=filtered_df, order=filtered_df['Sport'].value_counts().index[:15])
    st.pyplot()
