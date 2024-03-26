# importing used libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# set necessary functions
st.set_page_config(layout="wide") # to set page layout
df = pd.read_csv('dataset_olympics.csv') # to read dataset used
st.set_option('deprecation.showPyplotGlobalUse', False)

# data cleaning and preparation
df['Height'].fillna(df['Height'].mean(), inplace =True)
df['Weight'].fillna(df['Weight'].mean(), inplace =True)
df.dropna(subset=['Age'], inplace = True)
df.drop_duplicates(inplace = True)
df['Medal'].fillna('No Medal', inplace = True)

# side bar descriptions
st.sidebar.image('img.jpg', use_column_width=True)
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", ["HomePage", "Main Dashboard", "Dashboard 2","Medal Prediction"])

# homepage access
if page == 'HomePage':
    st.image('img.jpg', use_column_width=True) # inserting image for page
    st.title("Olympic Analytics Dashboard")
    st.markdown("---") # border between prints
    st.write("Welcome to the Olympic Analytics Dashboard!")
    st.write("Use the sidebar to navigate to different sections.")

# Dashboard 1: Visualizations 1-6
elif page == 'Main Dashboard': # condition when different dashboard is selected

        # Distribution of athletes by sex
        st.subheader('1. Distribution of Athletes by Sex')
        sport1 = st.multiselect('Select Sport', df['Sport'].unique(), key='sport1')
        year1 = st.multiselect('Select Year', df['Year'].unique(), key='year1')
        filter1 = df[df['Sport'].isin(sport1) & df['Year'].isin(year1)]
        if not filter1.empty:
            sns.countplot(x='Sex', data=filter1)
            plt.xticks(rotation=45)
            st.pyplot()
        st.markdown("---") # border between prints

        st.subheader('2. Age Distribution of Athletes')
        country1 = st.multiselect('Select Country', df['Team'].unique(), key='country1')
        age = st.slider("Select Age Range", min_value=df['Age'].min(), max_value=df['Age'].max(), value=(df['Age'].min(), df['Age'].max()))
        filter2 = df[df['Team'].isin(country1) & (df['Age'] >= age[0]) & (df['Weight'] <= age[1])]
        if not filter2.empty:
            sns.histplot(filter2['Age'], bins=20, kde=True)
            st.pyplot()
        st.markdown("---") # border between prints

        st.subheader('3. Weight Distribution of Athletes')
        country2 = st.multiselect('Select Country', df['Team'].unique(), key='country2')
        weight = st.slider("Select Weight Range", min_value=df['Weight'].min(), max_value=df['Weight'].max(), value=(df['Weight'].min(), df['Weight'].max()))
        filter3 = df[df['Team'].isin(country2) & ((df['Weight'] >= weight[0]) & (df['Weight'] <= weight[1]))]
        if not filter3.empty:
            sns.histplot(filter3['Weight'], bins=20, kde=True)
            st.pyplot()
        st.markdown("---") # border between prints

        st.subheader('4. Distribution of Athletes by Medal')
        medal1 = st.multiselect('Select Medal', ['Gold', 'Silver', 'Bronze', 'None'], key='medal1')
        country_filter_3 = st.multiselect('Select Country', df['Team'].unique(), key='country_filter_3')
        filter4 = df[df['Medal'].isin(medal1) & df['Team'].isin(country_filter_3)]
        if not filter4.empty:
            sns.countplot(x='Medal', data=filter4, order=['Gold', 'Silver', 'Bronze', 'None'])
            plt.xticks(rotation=45)
            st.pyplot()
        st.markdown("---") # border between prints

        st.subheader('5. Height Distribution of Athletes')
        year2 = st.multiselect('Select Year', df['Year'].unique(), key='year2')
        height = st.slider("Select Height Range", min_value=df['Height'].min(), max_value=df['Height'].max(), value=(df['Height'].min(), df['Height'].max()))
        filter5 = df[df['Year'].isin(year2) & ((df['Height'] >= height[0]) & (df['Height'] <= height[1]))]
        if not filter5.empty:
            sns.histplot(filter5['Height'], bins=20, kde=True)
            st.pyplot()
        st.markdown("---") # border between prints

        st.subheader('6. Number of Athletes per Year')
        season2 = st.multiselect('Select Season', df['Season'].unique(), key='season2')
        year3 = st.multiselect('Select Year', df['Year'].unique(), key='year_filter_3')
        filter6 = df[df['Season'].isin(season2) & df['Year'].isin(year3)]
        if not filter6.empty:
            sns.countplot(x='Year', data=filter6)
            plt.xticks(rotation=45)
            st.pyplot()
        st.markdown("---") # border between prints

# Dashboard 2: Visualizations 7-12
elif page == 'Dashboard 2':
        
    st.subheader('7. Distribution of Athletes by Season')
    sport2 = st.multiselect('Select Sport', df['Sport'].unique(), key='sport2')
    season3 = st.multiselect('Select Season', df['Season'].unique(), key='season3')
    filter7 = df[df['Sport'].isin(sport2) & df['Season'].isin(season3)]
    if not filter7.empty:
        sns.countplot(x='Season', data=filter7)
        plt.xticks(rotation=45)
        st.pyplot()
    st.markdown("---") # border between prints

    st.subheader('9. Distribution of Athletes by NOC')
    year3 = st.multiselect('Select Year', df['Year'].unique(), key='year3')
    season4 = st.multiselect('Select Season', df['Season'].unique(), key='season4')
    filtered_df_8 = df[df['Year'].isin(year3)&  df['Season'].isin(season4)]
    if not filtered_df_8.empty:
        sns.countplot(y='NOC', data=filtered_df_8, order=filtered_df_8['NOC'].value_counts().index[:15])
        st.pyplot()
    st.markdown("---") # border between prints
    
    st.subheader('11. Distribution of Medals by Year')
    year6 = st.multiselect('Select Year', df['Year'].unique(), key='year6')
    season6 = st.multiselect('Select Season', df['Season'].unique(), key='season6')
    filter11 = df[df['Year'].isin(year6) & df['Season'].isin(season6)]
    if not filter11.empty:
        plt.figure(figsize=(12, 8))
        sns.countplot(x='Year', hue='Medal', data=filter11)
        plt.xticks(rotation=45)
        st.pyplot()
    st.markdown("---") # border between prints

    st.subheader('8. Top 10 Participating Teams')
    medal2 = st.multiselect('Select Medal', ['Gold', 'Silver', 'Bronze', 'None'], key='medal2')
    sport3 = st.multiselect('Select Sport', df['Sport'].unique(), key='sport3')
    filter9 = df[df['Medal'].isin(medal2) & df['Sport'].isin(sport3)]
    if not filter9.empty:
        top_teams = filter9['Team'].value_counts().head(10)
        sns.barplot(x=top_teams.values, y=top_teams.index)
        plt.xlabel('Number of Athletes')
        plt.ylabel('Team')
        st.pyplot()
    st.markdown("---") # border between prints


    st.subheader('10. Distribution of Athletes by Sport')
    year4 = st.multiselect('Select Year', df['Year'].unique(), key='year4')
    sport4 = st.multiselect('Select Sport', df['Sport'].unique(), key='sport4')
    filter10 = df[df['Year'].isin(year4)  & df['Sport'].isin(sport4)]
    if not filter10.empty:
        sns.countplot(y='Sport', data=filter10, order=filter10['Sport'].value_counts().index[:15])
        plt.xlabel('Number of Athletes')
        plt.ylabel('Sport')
        st.pyplot()
    st.markdown("---") # border between prints

    st.subheader('12. Distribution of Medals by Sport')
    medal3 = st.multiselect('Select Medal', ['Gold', 'Silver', 'Bronze', 'None'], key='medal3')
    sport5 = st.multiselect('Select Sport', df['Sport'].unique(), key='sport5')
    filter12 = df[df['Medal'].isin(medal3) & df['Sport'].isin(sport5)]
    if not filter12.empty:
        sns.countplot(y='Sport', hue='Medal', data=filter12)
        st.pyplot()

elif page == 'Medal Prediction':

        st.markdown("---")
        st.subheader('1. Medal Prediction using Logistic Regression Model' )
        X = df[['Age', 'Height', 'Weight']]
        y = df['Medal'].apply(lambda x: 'Yes' if x != 'No Medal' else 'No')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='Yes')
        recall = recall_score(y_test, y_pred, pos_label='Yes')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Map numerical values to 'Yes' and 'No'
        labels = ['Yes', 'No']
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # side-by-side visualization 
        col1, col2 = st.columns(2)

        with col1:
            # Visualize Confusion Matrix
            st.subheader("Confusion Matrix")
            sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot()
            st.markdown("---")

        with col2:
        # Display evaluation metrics and predicted/actual results in a table
            st.subheader("Evaluation Metrics and Results")
            metrics_data = {
                'Metric': ['Accuracy', 'F1 Score', 'Recall', 'Predicted Result', 'Actual Result'],
                'Value': [accuracy, f1, recall, y_pred[0], y_test.iloc[0]]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)