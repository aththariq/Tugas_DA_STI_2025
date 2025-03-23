#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Principal Component Analysis (PCA) and Multi-Dimensional Scaling (MDS)
on the Iris dataset.

This script implements both algorithms from scratch and visualizes the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

def load_iris_data(file_path):
    """
    Load the Iris dataset from a CSV file.
    
    Args:
        file_path (str): Path to the Iris dataset file
        
    Returns:
        tuple: (X, y, feature_names, class_names)
    """
    # Define column names
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # Load the data
    data = pd.read_csv(file_path, header=None, names=column_names)
    
    # Extract features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Get unique class names
    class_names = np.unique(y)
    
    # Get feature names
    feature_names = column_names[:-1]
    
    return X, y, feature_names, class_names

def preprocess_data(X):
    """
    Preprocess the data by standardizing features.
    
    Args:
        X (numpy.ndarray): Input features
        
    Returns:
        numpy.ndarray: Standardized features
    """
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    return X_std

def implement_pca(X, n_components=2):
    """
    Implement Principal Component Analysis (PCA) from scratch.
    
    Args:
        X (numpy.ndarray): Input features
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (X_pca, explained_variance_ratio, eigenvectors, eigenvalues)
    """
    # Step 1: Calculate the mean of each feature
    mean_vector = np.mean(X, axis=0)
    
    # Step 2: Calculate the covariance matrix
    # Center the data
    X_centered = X - mean_vector
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Step 3: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Make sure eigenvalues and eigenvectors are real (not complex)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Step 4: Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top n_components eigenvectors
    eigenvectors_subset = eigenvectors[:, :n_components]
    
    # Step 6: Project data onto the new subspace
    X_pca = X_centered.dot(eigenvectors_subset)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance_ratio, eigenvectors_subset, eigenvalues

def implement_mds(X, n_components=2):
    """
    Implement Multi-Dimensional Scaling (MDS) from scratch.
    
    Args:
        X (numpy.ndarray): Input features
        n_components (int): Number of dimensions for the output
        
    Returns:
        numpy.ndarray: Projected data in lower dimensions
    """
    # Step 1: Calculate pairwise Euclidean distances
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            distance = np.sqrt(np.sum((X[i] - X[j])**2))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    # Step 2: Square the distances
    D_squared = distance_matrix ** 2
    
    # Step 3: Double centering
    # Create centering matrix
    centering_matrix = np.eye(n) - (1/n) * np.ones((n, n))
    
    # Apply double centering
    B = -0.5 * centering_matrix.dot(D_squared).dot(centering_matrix)
    
    # Step 4: Eigendecomposition of B
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top n_components eigenvalues and eigenvectors
    # Keep only positive eigenvalues
    positive_indices = eigenvalues > 0
    eigenvalues_positive = eigenvalues[positive_indices][:n_components]
    eigenvectors_positive = eigenvectors[:, positive_indices][:, :n_components]
    
    # Step 6: Compute the coordinates
    X_mds = eigenvectors_positive.dot(np.diag(np.sqrt(eigenvalues_positive)))
    
    return X_mds

def visualize_results(X_pca, X_mds, y, class_names, explained_variance_ratio):
    """
    Visualize the results of PCA and MDS.
    
    Args:
        X_pca (numpy.ndarray): PCA projected data
        X_mds (numpy.ndarray): MDS projected data
        y (numpy.ndarray): Target labels
        class_names (numpy.ndarray): Unique class names
        explained_variance_ratio (numpy.ndarray): Explained variance ratio for PCA
    """
    # Create a colormap
    colors = ['#FF0000', '#00FF00', '#0000FF']
    cmap = ListedColormap(colors[:len(class_names)])
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 7))
    
    # Plot PCA results
    plt.subplot(1, 2, 1)
    for i, cls in enumerate(class_names):
        plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], 
                   alpha=0.8, c=colors[i], label=cls, edgecolor='k')
    
    plt.title('PCA of Iris Dataset')
    plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot MDS results
    plt.subplot(1, 2, 2)
    for i, cls in enumerate(class_names):
        plt.scatter(X_mds[y == cls, 0], X_mds[y == cls, 1], 
                   alpha=0.8, c=colors[i], label=cls, edgecolor='k')
    
    plt.title('MDS of Iris Dataset')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_mds_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_separation(X_pca, X_mds, y, class_names):
    """
    Analyze the separation between classes in PCA and MDS projections.
    
    Args:
        X_pca (numpy.ndarray): PCA projected data
        X_mds (numpy.ndarray): MDS projected data
        y (numpy.ndarray): Target labels
        class_names (numpy.ndarray): Unique class names
    """
    # Calculate the centroid of each class
    centroids_pca = {}
    centroids_mds = {}
    
    for cls in class_names:
        centroids_pca[cls] = np.mean(X_pca[y == cls], axis=0)
        centroids_mds[cls] = np.mean(X_mds[y == cls], axis=0)
    
    # Calculate the average distance between centroids
    distances_pca = []
    distances_mds = []
    
    for i, cls1 in enumerate(class_names):
        for cls2 in class_names[i+1:]:
            dist_pca = np.linalg.norm(centroids_pca[cls1] - centroids_pca[cls2])
            dist_mds = np.linalg.norm(centroids_mds[cls1] - centroids_mds[cls2])
            
            distances_pca.append(dist_pca)
            distances_mds.append(dist_mds)
    
    avg_distance_pca = np.mean(distances_pca)
    avg_distance_mds = np.mean(distances_mds)
    
    print("\nClass Separation Analysis:")
    print(f"Average distance between class centroids in PCA: {avg_distance_pca:.4f}")
    print(f"Average distance between class centroids in MDS: {avg_distance_mds:.4f}")
    
    # Calculate within-class scatter
    scatter_pca = []
    scatter_mds = []
    
    for cls in class_names:
        scatter_pca.append(np.mean(np.linalg.norm(X_pca[y == cls] - centroids_pca[cls], axis=1)))
        scatter_mds.append(np.mean(np.linalg.norm(X_mds[y == cls] - centroids_mds[cls], axis=1)))
    
    avg_scatter_pca = np.mean(scatter_pca)
    avg_scatter_mds = np.mean(scatter_mds)
    
    print(f"Average within-class scatter in PCA: {avg_scatter_pca:.4f}")
    print(f"Average within-class scatter in MDS: {avg_scatter_mds:.4f}")
    
    # Calculate separation ratio (between-class / within-class)
    separation_ratio_pca = avg_distance_pca / avg_scatter_pca
    separation_ratio_mds = avg_distance_mds / avg_scatter_mds
    
    print(f"Separation ratio in PCA: {separation_ratio_pca:.4f}")
    print(f"Separation ratio in MDS: {separation_ratio_mds:.4f}")
    
    # Determine which method provides better separation
    if separation_ratio_pca > separation_ratio_mds:
        print("\nPCA provides better class separation for this dataset.")
    elif separation_ratio_mds > separation_ratio_pca:
        print("\nMDS provides better class separation for this dataset.")
    else:
        print("\nBoth methods provide similar class separation.")

def main():
    """
    Main function to run the PCA and MDS implementations.
    """
    # Load the Iris dataset
    file_path = "iris/iris.data"
    X, y, feature_names, class_names = load_iris_data(file_path)
    
    # Preprocess the data
    X_std = preprocess_data(X)
    
    # Implement PCA
    X_pca, explained_variance_ratio, eigenvectors, eigenvalues = implement_pca(X_std)
    
    # Implement MDS
    X_mds = implement_mds(X_std)
    
    # Print eigenvalues and explained variance
    print("PCA Eigenvalues:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"Eigenvalue {i+1}: {eigenvalue:.4f} (Explained Variance: {eigenvalue/sum(eigenvalues):.2%})")
    
    print("\nTotal Explained Variance by first 2 components: {:.2%}".format(
        sum(explained_variance_ratio)))
    
    # Visualize the results
    visualize_results(X_pca, X_mds, y, class_names, explained_variance_ratio)
    
    # Analyze class separation
    analyze_separation(X_pca, X_mds, y, class_names)
    
    # Print comparison summary
    print("\nComparison of PCA and MDS:")
    print("PCA:")
    print("  - Advantages: Preserves global variance, interpretable components")
    print("  - Limitations: Linear method, may not capture non-linear relationships")
    print("  - Interpretation: First component explains {:.2%} of variance, ".format(explained_variance_ratio[0]) +
          "second component explains {:.2%} of variance".format(explained_variance_ratio[1]))
    
    print("\nMDS:")
    print("  - Advantages: Preserves pairwise distances, can capture non-linear relationships")
    print("  - Limitations: Computationally intensive, dimensions not directly interpretable")
    print("  - Interpretation: Dimensions represent relative distances between samples")

if __name__ == "__main__":
    main()
