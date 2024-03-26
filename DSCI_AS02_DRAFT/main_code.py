# importing used libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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
page = st.sidebar.radio("Go to", ["HomePage", "Main Dashboard", "Dashboard 2","Model Prediction"])

# homepage access
if page == 'HomePage':
    st.image('img.jpg', use_column_width=True) # inserting image for page
    st.title("Olympic Analytics Dashboard")
    st.markdown("---") # border between prints
    st.subheader("Welcome to the Olympic Analytics Dashboard!")
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

    st.subheader('8. Distribution of Athletes by NOC')
    year3 = st.multiselect('Select Year', df['Year'].unique(), key='year3')
    season4 = st.multiselect('Select Season', df['Season'].unique(), key='season4')
    filtered_df_8 = df[df['Year'].isin(year3)&  df['Season'].isin(season4)]
    if not filtered_df_8.empty:
        sns.countplot(y='NOC', data=filtered_df_8, order=filtered_df_8['NOC'].value_counts().index[:15])
        st.pyplot()
    st.markdown("---") # border between prints
    
    st.subheader('9. Distribution of Medals by Year')
    year6 = st.multiselect('Select Year', df['Year'].unique(), key='year6')
    season6 = st.multiselect('Select Season', df['Season'].unique(), key='season6')
    filter11 = df[df['Year'].isin(year6) & df['Season'].isin(season6)]
    if not filter11.empty:
        plt.figure(figsize=(12, 8))
        sns.countplot(x='Year', hue='Medal', data=filter11)
        plt.xticks(rotation=45)
        st.pyplot()
    st.markdown("---") # border between prints

    st.subheader('10. Top 10 Participating Teams')
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


    st.subheader('11. Distribution of Athletes by Sport')
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

# Model Prediction section
elif page == 'Model Prediction':

# random forest classifier
    st.markdown("---")
    st.title('Medal Prediction using Random Forest Classifier :medal:')

    X = df[['Year', 'Age', 'Height', 'Weight']]
    y = df['Medal'].apply(lambda x: 1 if x != 'No Medal' else 0)

    # Train-Test split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    medal = RandomForestClassifier()
    medal.fit(X_train, y_train) 
    
    # Model evaluation
    y_pred = medal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Metrics evaluation
    st.write("Accuracy:", accuracy)
    st.write("F1 Score:", f1score)
    st.write("Recall:", recall)

    # Prediction
    age_input = st.slider("Select Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()))
    height_input = st.slider("Select Height", min_value=int(df['Height'].min()), max_value=int(df['Height'].max()), value=int(df['Height'].mean()))
    weight_input = st.slider("Select Weight", min_value=int(df['Weight'].min()), max_value=int(df['Weight'].max()), value=int(df['Weight'].mean()))
    
    year_input2 = st.selectbox("Select Year", options=df['Year'].unique())
    filtered_df = df[(df['Year'] == year_input2) & (df['Age'] == age_input) & (df['Height'] == height_input) & (df['Weight'] == weight_input)]

    if len(filtered_df) > 0:
        medal_prediction = medal.predict([[year_input2, age_input, height_input, weight_input]])

        # Get the actual medal color from the dataset for the specific athlete
        actual_medal_color = filtered_df[filtered_df['Medal'] != 'No Medal']['Medal'].iloc[0] if filtered_df['Medal'].any() else "No Medal"

        st.write("Predicted Medal:", "Yes" if medal_prediction[0] == 1 else "No")
        if medal_prediction[0] == 1:
            st.write("Medal Color:", actual_medal_color)
    else:
        st.write("No data available for the selected year and athlete's attributes.")

# Medal Prediction using Logistic Regression Model
st.markdown("---")
st.title('Medal Prediction using Logistic Regression Model :medal:' )

# inputs for age, height, and weight range
st.subheader("Model Training Parameters")
min_age2, max_age2 = st.slider("Select Age Range", df['Age'].min(), df['Age'].max(), (df['Age'].min(), df['Age'].max()))
min_height2, max_height2 = st.slider("Select Height Range", df['Height'].min(), df['Height'].max(), (df['Height'].min(), df['Height'].max()))
min_weight2, max_weight2 = st.slider("Select Weight Range", df['Weight'].min(), df['Weight'].max(), (df['Weight'].min(), df['Weight'].max()))
year_input2 = st.selectbox("Select Year", options=df['Year'].unique(), key="year_selectbox2")

# Filter the dataset based on selected ranges
filter_model2 = df[(df['Year'] == year_input2) &
                (df['Age'] >= min_age2) & (df['Age'] <= max_age2) &
                (df['Height'] >= min_height2) & (df['Height'] <= max_height2) &
                (df['Weight'] >= min_weight2) & (df['Weight'] <= max_weight2)]

X2 = filter_model2[['Age', 'Height', 'Weight']]
y2 = filter_model2['Medal'].apply(lambda x: 'Yes' if x != 'No Medal' else 'No')

# Train-test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Model training
model2 = LogisticRegression()
model2.fit(X_train2, y_train2)

# Model evaluation
y_pred2 = model2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
f12 = f1_score(y_test2, y_pred2, pos_label='Yes')
recall2 = recall_score(y_test2, y_pred2, pos_label='Yes')

# Confusion matrix
cm2 = confusion_matrix(y_test2, y_pred2)

# Map numerical values to 'Yes' and 'No'
labels2 = ['Yes', 'No']
cm_df2 = pd.DataFrame(cm2, index=labels2, columns=labels2)

# Get actual medals from the dataset for the test set
actual_medals2 = filter_model2.loc[y_test2.index, 'Medal']

# side-by-side visualization 
col1, col2 = st.columns(2)

with col1:
    # Visualize Confusion Matrix
    st.subheader("Confusion Matrix")
    sns.heatmap(cm_df2, annot=True, cmap='Blues', fmt='g', xticklabels=labels2, yticklabels=labels2)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()
    st.markdown("---")

with col2:
# Display evaluation metrics and predicted/actual results in a table
    st.subheader("Evaluation Metrics and Results")
    metrics_data2 = {
        'Metric': ['Accuracy', 'F1 Score', 'Recall', 'Predicted Result', 'Actual Result', 'Actual Medal'],
        'Value': [accuracy2, f12, recall2, y_pred2[0], y_test2.iloc[0], actual_medals2.iloc[0]]
    }
    metrics_df2 = pd.DataFrame(metrics_data2)
    st.table(metrics_df2)
