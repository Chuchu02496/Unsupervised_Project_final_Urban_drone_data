# Unsupervised Urban Scene Segmentation Project

## 1. Project Information

### Data Source (APA Citation)
> Bulent Siyah, Arial Semantic Segmentation Drone Dataset [Data set].Kaggle
> https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset, http://dronedataset.icg.tugraz.at
>
>
> Download the data set and transfer all the drone images into the data/original_images folder.

### Backstory:
Disaster relief operations often rely on rapid analysis of aerial imagery to identify accessible roads and damaged buildings. Analyzing these images manually is time-consuming and prone to error. I decided to create this unsupervised machine learning model to automate the segmentation of urban scenes, allowing for faster response times in critical situations. By grouping similar image patches based on color and texture, the model identifies distinct regions without the need for extensive labeled training data.

### Key Features:
-   **Custom K-Means Implementation**: A ground-up implementation of K-Means clustering using PyTorch `torch.cdist` for distance calculations.
-   **Advanced Feature Engineering**: Extracts both Color (HSL) and Texture (Local Binary Patterns) features to handle complex urban terrains.
-   **Hyperparameter Optimization**: Implements the Elbow Method to automatically determine the optimal number of clusters ($K$).
-   **Model Validation**: Compares a basic Color-Only model against the full Color+Texture model using Silhouette Scores to quantify improvements.

## 2. Data Source and Description
-   **Dataset**: High-resolution urban aerial images located in `dataset/original_images`.
-   **Preprocessing**: Images are tiled into non-overlapping $64 \times 64$ patches.
-   **Extracted Features**:
    -   **Color**: HSL (Hue, Saturation, Lightness) statistics. Hue is cyclical, so it is encoded as $sin(H)$ and $cos(H)$.
    -   **Texture**: Local Binary Patterns (LBP) histograms are computed to distinguish between smooth roads and textured vegetation/rubble.

## 3. Methodology: The Segmentation Engine
This project moves beyond standard library calls by implementing core algorithms from scratch to demonstrate a deep understanding of unsupervised learning principles.

### Feature Extraction
Logic is encapsulated in the `FeatureExtractor` class within `main_project.py`.
-   **RGB to HSL**: Converts raw pixel data to a perceptual color space more suitable for segmentation.
-   **LBP Computation**: Calculates texture descriptors by comparing each pixel to its neighbors, providing robustness against lighting variations.

### Clustering Algorithm
The clustering logic resides in the `CustomKMeans` class.
-   **Initialization**: Randomly selects initial centroids.
-   **Assignment**: Uses Euclidean distance (accelerated via PyTorch on CUDA/CPU) to assign patches to the nearest cluster.
-   **Update**: Recalculates centroids based on the mean of assigned patches until convergence.

## 4. Exploratory Data Analysis & Model Training
The pipeline performs a comprehensive analysis to ensure robust results.

### Exploratory Data Analysis (EDA)
The `perform_eda()` function generates visualizations saved to the `final_output/` directory:
-   **Boxplots**: Used to identify outliers in feature distributions (e.g., unusual lighting conditions).
-   **Correlation Matrix**: Visualizes relationships between color and texture features to ensure orthogonality and feature relevance.

### Model Comparison and Validation
To verify the design choices, the `compare_models()` function runs an A/B test:
1.  **Model A (Baseline)**: Uses only Color features.
2.  **Model B (Proposed)**: Uses Color + Texture features.
3.  **Metric**: Silhouette Score is calculated for both. The results are logged to `comparison_report.txt`, ensuring that the added complexity of LBP texture features yields a tangible performance gain.
