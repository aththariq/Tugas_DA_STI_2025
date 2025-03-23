# Principal Component Analysis (PCA) and Multi-Dimensional Scaling (MDS) on Iris Dataset

This project implements Principal Component Analysis (PCA) and Multi-Dimensional Scaling (MDS) algorithms from scratch on the Iris dataset.

## Overview

Both PCA and MDS are dimensionality reduction techniques that help visualize high-dimensional data in lower dimensions (typically 2D or 3D).

- **PCA**: Finds orthogonal axes (principal components) that maximize the variance in the data.
- **MDS**: Preserves pairwise distances between points in the original high-dimensional space.

## Implementation Details

The implementation includes:

1. **Data Loading and Preprocessing**:
   - Loading the Iris dataset
   - Standardizing features (zero mean, unit variance)

2. **PCA Implementation**:
   - Computing the mean vector
   - Computing the covariance matrix
   - Finding eigenvalues and eigenvectors
   - Selecting principal components based on eigenvalues
   - Projecting data onto the new subspace

3. **MDS Implementation**:
   - Computing pairwise Euclidean distances
   - Double centering the distance matrix
   - Eigendecomposition of the centered matrix
   - Computing coordinates in lower dimensions

4. **Visualization and Analysis**:
   - Visualizing PCA and MDS results
   - Analyzing class separation
   - Comparing the methods

## Results

### PCA Results

- First principal component explains 72.77% of variance
- Second principal component explains 23.03% of variance
- Together, they explain 95.80% of total variance

### Class Separation Analysis

- Average distance between class centroids in PCA: 2.7548
- Average within-class scatter in PCA: 0.8261
- Separation ratio in PCA: 3.3347

### Comparison

Both methods provide similar class separation for this dataset, with the same separation metrics. This is likely because the Iris dataset has a relatively simple structure that both methods can capture effectively.

## Usage

To run the implementation:

```bash
python pca_mds_iris.py
```

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn (only for StandardScaler)
- seaborn

## Files

- `pca_mds_iris.py`: Main implementation file
- `iris/iris.data`: Iris dataset
- `pca_mds_comparison.png`: Visualization of PCA and MDS results
