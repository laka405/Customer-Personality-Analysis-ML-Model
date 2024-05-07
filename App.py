import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

# Load the pre-trained KMeans model
with open('Customer_Personality_Analysis.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to predict the cluster for new data
def predict_cluster(data):
    return model.predict([data])[0]

# Main function to run the Streamlit app
def main():
    st.title('Customer Personality Analysis')
    st.sidebar.header('Enter Customer Information')

    # Input fields in the sidebar
    income = st.sidebar.number_input('Income', min_value=0)
    recency = st.sidebar.number_input('Recency', min_value=0)
    age = st.sidebar.number_input('Age', min_value=0)
    total_spendings = st.sidebar.number_input('Total Spendings', min_value=0)
    children = st.sidebar.number_input('Children', min_value=0)
    month_enrollment = st.sidebar.number_input('Month Enrollment', min_value=0)

    # Predicting cluster for the input data
    data = [income, recency, age, total_spendings, children, month_enrollment]
    cluster_label = predict_cluster(data)

    # Displaying cluster label
    st.subheader('Predicted Customer Cluster:')
    st.write(f'Cluster {cluster_label}')

    # You can calculate silhouette score using your own data if available
    # silhouette = silhouette_score(your_data, model.labels_)
    # st.write(f'Silhouette Score: {silhouette}')

# Run the main function
if __name__ == '__main__':
    main()
