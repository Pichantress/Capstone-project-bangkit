import math
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Shape of dataset -> {df.shape}")

    # Drop specified columns
    df = df.drop(['N', 'P', 'K', 'rainfall', 'ph'], axis=1)

    # Update 'temperature' and 'humidity' columns
    df['temperature'] = df['temperature'].apply(math.ceil).astype(int)
    df['humidity'] = df['humidity'].apply(math.ceil).astype(int)

    # Display the dataframe after rounding values
    print(df)

    return df

def remove_outliers_and_clean(df):
    # Calculate quartiles and IQR
    Q1 = df['temperature'].quantile(0.25)
    Q3 = df['temperature'].quantile(0.75)
    IQR = Q3 - Q1
    boxplot_min = Q1 - 1.5 * IQR
    boxplot_max = Q3 + 1.5 * IQR

    # Filter outliers
    filter_min = df['temperature'] < boxplot_min
    filter_max = df['temperature'] > boxplot_max
    data_df = df[~(filter_min | filter_max)]

    # Display information after removing outliers
    print('Q1:\n', Q1)
    print('\nQ3:\n', Q3)
    print('\nIQR:\n', IQR)
    print('\nMin:\n', boxplot_min)
    print('\nMax:\n', boxplot_max)
    print('\nShape after removing outliers:', data_df.shape)
    print('\nNull values after removing outliers:')
    print(data_df.isnull().sum())

    # Cleaned DataFrame
    df_clean = data_df.copy()
    df_clean.columns = ['temperature', 'humidity', 'label']
    print("Jumlah sampel sebelum SMOTE:")
    print(df_clean['label'].value_counts())
    print("=============================")

    # Separate features and target
    X = df_clean[['temperature', 'humidity']]
    y = df_clean['label']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Form a new dataset after SMOTE
    dataset = pd.DataFrame(X_res, columns=['temperature', 'humidity'])
    dataset['label'] = y_res

    # Display the number of samples after SMOTE
    print("\nJumlah sampel setelah SMOTE:")
    print(y_res.value_counts())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=10)

    print('training dataset:')
    print(X_train.shape)
    print(y_train.shape)
    print()
    print('testing dataset:')
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test, dataset

def perform_kmeans_and_plot_silhouette(X_train, k_range):
    # Drop the 'label' column if it exists
    if 'label' in X_train.columns:
        df_kmeans = X_train.drop(['label'], axis=1)
    else:
        df_kmeans = X_train.copy()

    # Store silhouette scores for each number of clusters
    silhouette_scores = []

    # Perform clustering and compute silhouette score for each number of clusters
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_kmeans)
        labels = kmeans.labels_
        score = silhouette_score(df_kmeans, labels)
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.plot(k_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method for Determining Optimal Number of Clusters')
    plt.show()

    return df_kmeans

def perform_kmeans_clustering(X_train, n_clusters):
    # Initialize K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Train the model with plant data
    kmeans.fit(X_train)

    # Predict clusters for plant data
    labels = kmeans.labels_

    # Add 'cluster' column to the DataFrame
    df_kmeans = pd.DataFrame(X_train, columns=['temperature', 'humidity'])
    df_kmeans['cluster'] = labels
    df_kmeans['label'] = y_train

    # Print cluster centers
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)

    # Print clusters for each plant data
    print("\nClusters for Each Plant Data:")
    print(df_kmeans)

    return df_kmeans, kmeans

def visualize_clusters(X_train, labels, kmeans):
    # Predict cluster labels
    y_kmeans = kmeans.predict(X_train)

    # Scatter plot
    plt.scatter(X_train[y_kmeans == 0]['temperature'], X_train[y_kmeans == 0]['humidity'], c='red', label='Cluster 0')
    plt.scatter(X_train[y_kmeans == 1]['temperature'], X_train[y_kmeans == 1]['humidity'], c='blue', label='Cluster 1')
    plt.scatter(X_train[y_kmeans == 2]['temperature'], X_train[y_kmeans == 2]['humidity'], c='green', label='Cluster 2')

    # Plot centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', label='Centroids')

    # Add labels to the x and y axis
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    # Save the KMeans model
    filename = f'kmeans_model.pickle'
    pickle.dump(kmeans, open(filename, "wb"))
    print(f'KMeans model saved successfully with the name {filename}')

def export_clusters_as_csv(dataset_clusters):
    # Separate the dataframe for each cluster
    cluster_dfs = {}
    for cluster_label in dataset_clusters['cluster'].unique():
        cluster_dfs[cluster_label] = dataset_clusters[dataset_clusters['cluster'] == cluster_label].drop('cluster', axis=1, errors='ignore')

    # Print labels for each cluster
    for cluster_label, cluster_df in cluster_dfs.items():
        print(f"Cluster {cluster_label}:")
        if 'label' in cluster_df.columns:
            labels = cluster_df['label'].unique()
            print("Labels:", labels)
        else:
            print("No 'label' column in the cluster.")
        print("=============================")

    # Export each cluster as a new dataframe
    for cluster_label in dataset_clusters['cluster'].unique():
        cluster_df = dataset_clusters[dataset_clusters['cluster'] == cluster_label]
        cluster_df.to_csv(f'cluster_{cluster_label}.csv', index=False)

def load_and_classify_clusters(cluster_data):
    highest_accuracy = 0.0
    best_cluster = None

    for cluster_name, cluster_df in cluster_data.items():
        # Extract features and labels from the DataFrame
        X = cluster_df.drop('label', axis=1)
        y = cluster_df['label']

        # Create a Decision Tree Classifier model
        model = DecisionTreeClassifier()

        # Train the model
        model.fit(X, y)

        # Save the model for each cluster
        filename = f'model_{cluster_name}.pickle'
        pickle.dump(model, open(filename, "wb"))
        print(f'Model {filename} saved successfully')

        # Make predictions using the model
        y_pred = model.predict(X)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y, y_pred)
        print(accuracy)

        # Compare accuracy with the highest value so far
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_cluster = cluster_name

    # Return the DataFrame with the highest accuracy
    return cluster_data[best_cluster]

# Example usage:
file_path = './Dataset/Crop_recommendation.csv'
original_df = load_and_preprocess_data(file_path)

# Remove outliers and clean the dataset
X_train, X_test, y_train, y_test, dataset_after_smote = remove_outliers_and_clean(original_df)

# Perform K-means clustering and plot silhouette scores
k_range = range(2, 10)
df_kmeans = perform_kmeans_and_plot_silhouette(X_train, k_range)

# Perform K-means clustering with the desired number of clusters
n_clusters = 3
df_kmeans, kmeans_model = perform_kmeans_clustering(X_train, n_clusters)

# Visualize clusters and save the KMeans model
visualize_clusters(X_train, df_kmeans['cluster'], kmeans_model)

# Export clusters as CSV
export_clusters_as_csv(df_kmeans)

# Load and classify clusters
dataframes = {'cluster_0': pd.read_csv('cluster_0.csv'), 'cluster_1': pd.read_csv('cluster_1.csv'), 'cluster_2': pd.read_csv('cluster_2.csv')}
best_cluster_df = load_and_classify_clusters(dataframes)

print("DataFrame with the highest accuracy:")
