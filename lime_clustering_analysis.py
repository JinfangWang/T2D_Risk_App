# Updated 2024/10/13 8:13
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import lime.lime_tabular
import matplotlib.pyplot as plt
from scipy.stats import entropy, f
import pprint
import pickle
import os
from tensorflow.keras import layers, models
import shap

class LimeClusteringAnalysis:
    def __init__(self, data, target_column, explainall=False, pca_components=8, model=None, random_state=42, test_size=0.2):
        self.data = data
        self.target_column = target_column
        self.pca_components = pca_components
        self.random_state = random_state
        self.X = self.data.drop(columns=[self.target_column])
        self.Y = self.data[self.target_column]
        
        # if explainall=False (default), train-test split is performed.
        # if explainall=True, no train-test split is performed.
        if not explainall:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.X, self.Y, test_size=test_size, random_state=random_state
            )
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.X.copy(), self.X.copy(), self.Y.copy(), self.Y.copy()  # Use the full dataset for explanation
        
        # Extract '加入者id' for matching with disease data
        self.ID = self.X['加入者id']
        self.ID_train = self.X_train['加入者id']
        self.ID_test = self.X_test['加入者id']

        # Exclude '加入者id' from the feature sets
        self.X = self.X.drop(columns=['加入者id'])
        self.X_train = self.X_train.drop(columns=['加入者id'])
        self.X_test = self.X_test.drop(columns=['加入者id'])

        # Initialize a model, defaulting to RandomForest if none is provided
        self.model = model if model else RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.lime_importances_df = None
        self.pca_df = None
        self.clusters_original = None
        self.clusters_pca = None
        self.clusters_features = None
        self.risks_original = None
        self.risks_pca = None
        self.risks_features = None
        self.cluster_risk_order_original = None # Added for debugging 12/4

    def train_model(self):
        """Train the classifier on the training data."""
        # Train the model
        self.model.fit(self.X_train, self.Y_train)
        accuracy = self.model.score(self.X_test, self.Y_test)
        print(f"Test Accuracy: {accuracy:.2f}")
    
    def train_logistic_regression(self, max_iter=1000, solver='lbfgs'):
        """
        Train a logistic regression model on the training data.
        
        Parameters:
        - max_iter (int): Maximum number of iterations for solver to converge. Default is 1000.
        - solver (str): Algorithm to use in the optimization problem ('lbfgs', 'liblinear', etc.). Default is 'lbfgs'.
        """
        
        # Step 1: Standardize the training and test data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Step 2: Define and train the logistic regression model
        logit_model = LogisticRegression(
            random_state=self.random_state, max_iter=max_iter, solver=solver
        )
        logit_model.fit(X_train_scaled, self.Y_train)
    
        # Step 3: Evaluate the model on the test set
        accuracy = logit_model.score(X_test_scaled, self.Y_test)
        print(f"Test accuracy of logistic model: {accuracy:.2f}")
    
        # Step 4: Store the model for future use
        self.logistic_model = logit_model
    
    def generate_lime_importances(self, save_path=None, load_path=None):
        """
        Generate LIME contribution weights for each sample in the test set.
        Parameters:
        - save_path (str): Path to save the LIME importance results after generation.
        - load_path (str): Path to load previously saved LIME importance results.
        """

        # Check if we are loading previously saved importances
        if load_path and os.path.exists(load_path):
            with open(load_path, 'rb') as file:
                self.lime_importances_df = pickle.load(file)
            print(f"LIME importances loaded from {load_path}")
            return
    
        # Otherwise, generate LIME importances
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=['No T2D', 'T2D'],
            mode='classification'
        )
        
        def predict_with_feature_names(x):
            x_df = pd.DataFrame(x, columns=self.X_train.columns)
            return self.model.predict_proba(x_df)
        
        lime_importances = []

        # Generate LIME explanations for the test set
        for i in range(len(self.X_test)):
            data_row = self.X_test.iloc[i].values
            exp = explainer.explain_instance(
                data_row, 
                predict_fn=predict_with_feature_names, 
                num_features=len(self.X_train.columns)
            )
            importance_dict = dict(exp.as_list())

            # Create a vector of importance scores in the original feature order
            importance_vector = [
                sum(coef for feat, coef in importance_dict.items() if feature in feat)
                for feature in self.X_train.columns
            ]
            lime_importances.append(importance_vector)

        # Store LIME importances DataFrame
        self.lime_importances_df = pd.DataFrame(lime_importances, columns=self.X_train.columns)
        print("LIME importances generated and stored.")

        # Save LIME importances if a save path is provided
        if save_path:
            with open(save_path, 'wb') as file:
                pickle.dump(self.lime_importances_df, file)
            print(f"LIME importances saved to {save_path}")

    def perform_pca(self):
        """Perform PCA on the LIME importances."""
        if self.lime_importances_df is None:
            raise ValueError(
                "LIME importances not generated. Please run `generate_lime_importances()` first."
            )
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.lime_importances_df)
        
        # Apply PCA
        pca = PCA(n_components=self.pca_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Store PCA DataFrame
        self.pca_df = pd.DataFrame(
            pca_result, columns=[f'PC{i+1}' for i in range(self.pca_components)]
        )
        #self.pca_df['T2D'] = self.Y_test.values
        print("PCA performed and stored.")

    def perform_clustering(self, n_clusters):
        """Perform K-Means clustering on the LIME importance weights and PCA-transformed data."""

        if self.lime_importances_df is None or self.pca_df is None:
            raise ValueError(
                "LIME importances or PCA results not available. Please run `generate_lime_importances()` and `perform_pca()` first."
            )
        
        # Scaling Data # modified 2024/10/14
        self.scaler_lime = StandardScaler()

        # Clustering with original LIME importances (without PCA)
        # Scale only the original LIME importances, without adding cluster columns
        scaled_lime_importances = self.scaler_lime.fit_transform(self.lime_importances_df.drop(columns=['Cluster_Original'], errors='ignore'))  # Exclude cluster columns here
        self.kmeans_original = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.clusters_original = self.kmeans_original.fit_predict(scaled_lime_importances)
        self.lime_importances_df['Cluster_Original'] = self.clusters_original

        # Clustering with PCA-transformed data
        self.scaler_pca = StandardScaler() # modified 2024/10/14
        scaled_pca = self.scaler_pca.fit_transform(self.pca_df.drop(columns=['Cluster_PCA'], errors='ignore'))  # Exclude cluster columns here
        self.kmeans_pca = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.clusters_pca = self.kmeans_pca.fit_predict(scaled_pca)
        self.pca_df['Cluster_PCA'] = self.clusters_pca

        # Clustering with original features
        self.scaler_features = StandardScaler() # modified 2024/10/14
        # Ensure you drop the 'Cluster_Features' column, so scaler matches the features used in fit
        scaled_X_test = self.scaler_features.fit_transform(self.X_test.drop(columns=['Cluster_Features'], errors='ignore'))
        self.kmeans_features = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.clusters_features = self.kmeans_features.fit_predict(scaled_X_test)
        self.X_test['Cluster_Features'] = self.clusters_features
        
        print(f"Clustering performed with {n_clusters} clusters and stored.")

    def compute_and_order_cluster_risks(self):
        """
        - Compute T2D risk for each centroid and store risks for each clustering method.
        - Compute T2D risk for each individual data point and store risks for each clustering method.
        - Order the clusters based on T2D risks for consistent comparison.
       """
        if self.clusters_original is None or self.clusters_pca is None or self.clusters_features is None:
            raise ValueError("Clusters not available. Please run `perform_clustering()` first.")

        # Function to calculate risk for each cluster centroid
        def calculate_risk_for_centroid(scaled_data, cluster_labels):
            # Take out the columns of scaled_data with the columns of self.X_train (new changes)
            scaled_data_df = scaled_data[self.X_train.columns]

            # Calculate cluster centroids by grouping the data by cluster labels and taking the mean
            cluster_centroids = scaled_data_df.groupby(cluster_labels).mean()

            # Predict T2D risk using the trained logistic model
            risks = self.logistic_model.predict_proba(cluster_centroids.values)[:, 1]  # Probability of T2D (assuming class 1 is T2D)
            return risks

            # Function to calculate risk for each data point in each cluster
        def calculate_risk_for_individuals(scaled_data):
                """Predict T2D probability for each individual data point."""
                # Take out the columns of scaled_data with the columns of self.X_train (new changes)
                scaled_data_df = scaled_data[self.X_train.columns]

                # Predict T2D risk for each individual data point
                risks = self.logistic_model.predict_proba(scaled_data_df.values)[:, 1]  # Probability of T2D (assuming class 1 is T2D)
                return risks
    
        # Step 1: Standardize the original test features for consistent centroid calculation      
        scaled_X_test = self.X_test[self.X_train.columns].copy()
        
        # Remove any columns that were added after the scaler was fitted
        if 'Cluster_Features' in scaled_X_test.columns:
            scaled_X_test = scaled_X_test.drop(columns=['Cluster_Features'])
        
        # Apply the scaling transformation
        
        scaled_X_test = self.scaler.transform(scaled_X_test)  # Transform using the fitted scaler
        scaled_X_test_df = pd.DataFrame(scaled_X_test, columns=self.X_train.columns)

        # Step 2: Calculate risks for the centroids of the three clustering methods

        self.risks_original = calculate_risk_for_centroid(scaled_X_test_df, self.clusters_original)
        self.risks_pca = calculate_risk_for_centroid(scaled_X_test_df, self.clusters_pca)
        self.risks_features = calculate_risk_for_centroid(scaled_X_test_df, self.clusters_features)

        # Step 3: Calculate logistic probabilities for each individual data point
        self.data_point_risks_original = calculate_risk_for_individuals(scaled_X_test_df)
        self.data_point_risks_pca = calculate_risk_for_individuals(scaled_X_test_df)
        self.data_point_risks_features = calculate_risk_for_individuals(scaled_X_test_df)

        # Step 4: Order the clusters according to T2D risks (from lowest to highest)
        ordered_clusters_original = np.argsort(self.risks_original)
        ordered_clusters_pca = np.argsort(self.risks_pca)
        ordered_clusters_features = np.argsort(self.risks_features)

        # Step 5: Create label mapping from order to original labels
        self.cluster_risk_order_original = {i: rank for rank, i in enumerate(ordered_clusters_original)}
        self.cluster_risk_order_pca = {i: rank for rank, i in enumerate(ordered_clusters_pca)}
        self.cluster_risk_order_features = {i: rank for rank, i in enumerate(ordered_clusters_features)}

        # Step 6: Update the cluster labels to ensure consistency according to risk ordering
        # Reset the index of the X_test DataFrame is essential for alignment!
        self.X_test = self.X_test.reset_index(drop=True)
        # Verify that indices are aligned
        if not self.X_test.index.equals(self.lime_importances_df.index):
              print("Warning: Index misalignment detected between X_test and lime_importances_df")
        if not self.X_test.index.equals(self.pca_df.index):
              print("Warning: Index misalignment detected between X_test and pca_df")  

        # Update cluster labels using the mapping created from ordering based on risk
        # Update the cluster labels in the X_test DataFrame (new changes)
        self.lime_importances_df['Cluster_Original_Ordered'] = self.lime_importances_df['Cluster_Original'].map(self.cluster_risk_order_original)
        self.pca_df['Cluster_PCA_Ordered'] = self.pca_df['Cluster_PCA'].map(self.cluster_risk_order_pca)
        self.X_test['Cluster_Features_Ordered'] = self.X_test['Cluster_Features'].map(self.cluster_risk_order_features)
    
      
        # Step 7: Create the X_test_heatmap DataFrame for visualization
        
        # Copy self.X_test with only columns from self.X_train
        X_test_heatmap = self.X_test[self.X_train.columns].copy()

        # Standardize the data in X_test_heatmap
        scaler = StandardScaler()
        X_test_heatmap_scaled = scaler.fit_transform(X_test_heatmap)
        X_test_heatmap_scaled = pd.DataFrame(X_test_heatmap_scaled, columns=X_test_heatmap.columns)

        # Add columns to X_test_heatmap_scaled with the ordered cluster labels for visualization
        X_test_heatmap_scaled['Cluster_LIME_Ordered'] = self.lime_importances_df['Cluster_Original_Ordered']
        X_test_heatmap_scaled['Cluster_PCA_Ordered'] = self.pca_df['Cluster_PCA_Ordered']
        X_test_heatmap_scaled['Cluster_Features_Ordered'] = self.X_test['Cluster_Features_Ordered']

        print("\nCluster labels updated based on T2D risks.")
        print("T2D risks for centroids and individual data points computed and stored.")
        
        # Return the standardized X_test_heatmap and the cluster risk orderings
        return X_test_heatmap_scaled, self.cluster_risk_order_original, self.cluster_risk_order_pca, self.cluster_risk_order_features


    def compute_between_within_variances(self):
        """
        - Compute the between-cluster variance and within-cluster variance of the T2D risks
          for each clustering method (original, PCA, and features), 
        - Compute a scalar F-statistic (ratio of between-cluster variance to within-cluster variance) and its 
        corresponding p-value.
       """
        if self.data_point_risks_original is None or self.data_point_risks_pca is None or self.data_point_risks_features is None:
            raise ValueError("Individual T2D risks not computed. Please run `compute_and_order_cluster_risks()` first.")
        
        num_clusters_original = len(np.unique(self.clusters_original))
        num_clusters_pca = len(np.unique(self.clusters_pca))
        num_clusters_features = len(np.unique(self.clusters_features))

        total_data_points_original = len(self.data_point_risks_original)
        total_data_points_pca = len(self.data_point_risks_pca)
        total_data_points_features = len(self.data_point_risks_features)

        def calculate_between_within_variance(risks, cluster_labels):
            """Helper function to compute between and within variance for a clustering method."""
            overall_mean_risk = np.mean(risks)  # Overall mean risk across all data points
            unique_clusters = np.unique(cluster_labels)
            
            between_variance = 0
            within_variance = 0
            total_data_points = len(risks)
            
            for cluster in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster)[0]  # Indices of points in this cluster
                cluster_risks = risks[cluster_indices]
                cluster_mean_risk = np.mean(cluster_risks)
                cluster_size = len(cluster_risks)
                
                # Between-cluster variance component
                between_variance += cluster_size * (cluster_mean_risk - overall_mean_risk) ** 2
                # Within-cluster variance component
                within_variance += np.sum((cluster_risks - cluster_mean_risk) ** 2)
        
                # Normalize the variances
                between_variance /= len(unique_clusters)  # Number of clusters
                within_variance /= total_data_points  # Number of data points

                return between_variance, within_variance

        def compute_f_statistic_and_p_value(between_var, within_var, num_clusters, num_data_points):
            """Compute the F-statistic and corresponding p-value."""
            # Degrees of freedom
            df_between = num_clusters - 1  # Between-group degrees of freedom
            df_within = num_data_points - num_clusters  # Within-group degrees of freedom
        
            # F-statistic
            f_statistic = between_var / within_var
        
            # Compute the p-value from the F-distribution survival function
            p_value = f.sf(f_statistic, df_between, df_within)
        
            return f_statistic, p_value
    
        # Compute between and within variance for original LIME clusters
        between_variance_original, within_variance_original = calculate_between_within_variance(
            self.data_point_risks_original, self.clusters_original
        )

        f_statistic_original, p_value_original = compute_f_statistic_and_p_value(
            between_variance_original, within_variance_original, num_clusters_original, total_data_points_original
        )

        # Compute between and within variance for PCA clusters
        between_variance_pca, within_variance_pca = calculate_between_within_variance(
            self.data_point_risks_pca, self.clusters_pca
        )

        f_statistic_pca, p_value_pca = compute_f_statistic_and_p_value(
            between_variance_pca, within_variance_pca, num_clusters_pca, total_data_points_pca
        )

        # Compute between and within variance for feature clusters
        between_variance_features, within_variance_features = calculate_between_within_variance(
            self.data_point_risks_features, self.clusters_features
        )

        f_statistic_features, p_value_features = compute_f_statistic_and_p_value(
            between_variance_features, within_variance_features, num_clusters_features, total_data_points_features
        )
    
        result = {
            'Original_Clustering': {
                #'Between_Cluster_Variance': between_variance_original,
                #'Within_Cluster_Variance': within_variance_original,
                'F_Statistic': f_statistic_original,
                'P_Value': f"{p_value_original:.5e}"},
            'PCA_Clustering': {
                #'Between_Cluster_Variance': between_variance_pca,
                #'Within_Cluster_Variance': within_variance_pca,
                'F_Statistic': f_statistic_pca,
                'P_Value': f"{p_value_pca:.5e}"},
            'Feature_Clustering': {
                #'Between_Cluster_Variance': between_variance_features,
                #'Within_Cluster_Variance': within_variance_features,
                'F_Statistic': f_statistic_features,
                'P_Value': f"{p_value_features:.5e}"}
                }

        # Pretty-print the result
        pprint.pprint(result)

    def compute_entropy(self):
        """
        Compute the entropy based on the normalized T2D risks of each cluster.
        This entropy measures the uncertainty in T2D risks across clusters.
        """
        if self.risks_original is None or self.risks_pca is None or self.risks_features is None:
            raise ValueError("T2D risks not computed. Please run `compute_and_order_cluster_risks()` first.")

        def calculate_normalized_entropy(risks):
            """Helper function to compute entropy from a list of T2D risks."""
            # Normalize risks to form a probability distribution
            total_risk = np.sum(risks)
            normalized_risks = risks / total_risk

            # Compute entropy
            return entropy(normalized_risks)

        # Calculate entropies for the three clustering methods
        entropy_original = calculate_normalized_entropy(self.risks_original)
        entropy_pca = calculate_normalized_entropy(self.risks_pca)
        entropy_features = calculate_normalized_entropy(self.risks_features)
        
        result =  {
            'Entropy_Original': f"{entropy_original:.3f}",
            'Entropy_PCA': f"{entropy_pca:.3f}",
            'Entropy_Features': f"{entropy_features:.3f}"
        }

        # Pretty-print the result
        pprint.pprint(result)

    def compare_clustering_methods(self, save_path=None):
        """Compare clustering using original features, LIME importance without PCA, and PCA-transformed LIME importance."""
        if self.clusters_pca is None or self.clusters_features is None:
            raise ValueError("Clusters not available. Please run `perform_clustering()` first.")
        
        plt.figure(figsize=(6, 13.5))

        # Plot clusters for original LIME importances in PCA space
        plt.subplot(3, 1, 1)
        plt.scatter(
            self.pca_df['PC1'], self.pca_df['PC2'],
            c=self.lime_importances_df['Cluster_Original'],
            cmap='coolwarm', alpha=0.5, s=10
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Clusters Using Original LIME Importances')
        plt.colorbar(label='Cluster Label (Original LIME)')

        # Plot clusters for PCA-transformed LIME importances
        plt.subplot(3, 1, 2)
        plt.scatter(
            self.pca_df['PC1'], self.pca_df['PC2'],
            c=self.pca_df['Cluster_PCA'],
            cmap='viridis', alpha=0.5, s=10
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Clusters Using PCA-Transformed LIME Importances')
        plt.colorbar(label='Cluster Label (PCA)')

        # Plot clusters for original features in PCA space
        plt.subplot(3, 1, 3)
        plt.scatter(
            self.pca_df['PC1'], self.pca_df['PC2'],
            c=self.clusters_features,
            cmap='plasma', alpha=0.5, s=10
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Clusters Using Original Features')
        plt.colorbar(label='Cluster Label (Original Features)')

        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()
        
    def visualize_feature_association(self, X_test_heatmap_scaled, save_path=None, include='all', title=None):
        """
        Visualize the degree of association between each feature and each cluster for different clustering methods.

       Parameters:
       - X_test_heatmap_scaled: DataFrame containing the standardized features and ordered cluster labels.
       - save_path (str, optional): Path to save the output figure. Defaults to None.
       - include (str, optional): Determines which heatmap to plot. Can be 'features', 'lime', 'pca', or 'all'. Defaults to 'all'.
       - title (str, optional): Title for the entire figure or individual plots if provided. Defaults to None.
       """
        if self.clusters_original is None or self.clusters_pca is None or self.clusters_features is None:
            raise ValueError("Clusters not available. Please run `perform_clustering()` first.")
    
        # Define columns to exclude (cluster labels that are not relevant for each heatmap)
        exclude_columns_features = ['Cluster_LIME_Ordered', 'Cluster_PCA_Ordered']
        exclude_columns_lime = ['Cluster_Features_Ordered', 'Cluster_PCA_Ordered']
        exclude_columns_pca = ['Cluster_LIME_Ordered', 'Cluster_Features_Ordered']
    
        # Plot the heatmaps based on the include parameter
        if include == 'features' or include == 'all':
            # Plot heatmap for Original Features
            plt.figure(figsize=(6, 4))
        
            # Drop irrelevant columns and group by 'Cluster_Features_Ordered'
            X_test_heatmap_features = X_test_heatmap_scaled.drop(columns=exclude_columns_features)
            cluster_means_features = X_test_heatmap_features.groupby('Cluster_Features_Ordered').mean().T

            # Print number of samples in each cluster
            cluster_counts_features = X_test_heatmap_features['Cluster_Features_Ordered'].value_counts().sort_index()
            print("Number of samples in each cluster (Original Features Ordered):")
            print(cluster_counts_features)

            sns.heatmap(
                cluster_means_features, annot=True, cmap='coolwarm',
                linewidths=0.5, fmt=".2f", cbar_kws={'label': 'Z-Score'},
                vmin=-2, vmax=2, annot_kws={"size": 5}
                )
            plt.xlabel('Clusters Based on Original Features (Ordered)')
            plt.yticks(rotation=0, fontsize=8)
            if title:
                plt.title(f"{title}")
                plt.tight_layout()

            # Save the figure if save_path is provided
                if save_path and include == 'features':
                    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
                    print(f"Figure saved to {save_path}")

            plt.show()

        if include == 'lime' or include == 'all':
           # Plot heatmap for LIME Importance Without PCA
           plt.figure(figsize=(6, 4))

           # Drop irrelevant columns and group by 'Cluster_LIME_Ordered'
           X_test_heatmap_lime = X_test_heatmap_scaled.drop(columns=exclude_columns_lime)
           cluster_means_lime = X_test_heatmap_lime.groupby('Cluster_LIME_Ordered').mean().T

           # Print number of samples in each cluster
           cluster_counts_lime = X_test_heatmap_lime['Cluster_LIME_Ordered'].value_counts().sort_index()
           print("Number of samples in each cluster (LIME Importances Ordered):")
           print(cluster_counts_lime)

           sns.heatmap(
               cluster_means_lime, annot=True, cmap='coolwarm',
               linewidths=0.5, fmt=".2f", cbar_kws={'label': 'Z-Score'},
               vmin=-2, vmax=2, annot_kws={"size": 5}
               )
           plt.xlabel('Clusters Based on LIME Importances (Ordered)')
           plt.yticks(rotation=0, fontsize=8)
           if title:
               plt.title(f"{title}")
               plt.tight_layout()

           # Save the figure if save_path is provided
           if save_path and include == 'lime':
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

           plt.show()

        if include == 'pca' or include == 'all':
          # Plot heatmap for LIME Importance With PCA
          plt.figure(figsize=(6, 4))

          # Drop irrelevant columns and group by 'Cluster_PCA_Ordered'
          X_test_heatmap_pca = X_test_heatmap_scaled.drop(columns=exclude_columns_pca)
          cluster_means_pca = X_test_heatmap_pca.groupby('Cluster_PCA_Ordered').mean().T

          # Print number of samples in each cluster
          cluster_counts_pca = X_test_heatmap_pca['Cluster_PCA_Ordered'].value_counts().sort_index()
          print("Number of samples in each cluster (PCA Transformed LIME Importances Ordered):")
          print(cluster_counts_pca)

          sns.heatmap(
              cluster_means_pca, annot=True, cmap='coolwarm',
              linewidths=0.5, fmt=".2f", cbar_kws={'label': 'Z-Score'},
              vmin=-2, vmax=2, annot_kws={"size": 5}
              )
          plt.xlabel('Clusters Based on PCA-Transformed LIME Importances')
          plt.yticks(rotation=0, fontsize=8)
          if title:
              plt.title(f"{title}")
              plt.tight_layout()

          # Save the figure if save_path is provided
          if save_path and include == 'pca':
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

          plt.show()
    
    def compute_cpr_score(self):
        """
        Compute the Cluster Predictive Relevance (CPR) Score for each clustering method.
        The CPR score is the ratio of between-cluster variance to within-cluster variance 
        in terms of predicted T2D risk.
        """
        if self.data_point_risks_original is None or self.data_point_risks_pca is None or self.data_point_risks_features is None:
            raise ValueError("Individual T2D risks not computed. Please run `compute_and_order_cluster_risks()` first.")
    
        def calculate_within_cluster_variance(risks, cluster_labels):
            """Helper function to compute within-cluster variance."""
            unique_clusters = np.unique(cluster_labels)
            within_variance = 0

            for cluster in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster)[0]  # Indices of points in this cluster
                cluster_risks = risks[cluster_indices]
                cluster_mean_risk = np.mean(cluster_risks)
            
                # Within-cluster variance component
                within_variance += np.sum((cluster_risks - cluster_mean_risk) ** 2)
        
                # Normalize by the total number of data points
                within_variance /= len(risks)
                return within_variance

        def calculate_between_cluster_variance(risks, cluster_labels):
            """Helper function to compute between-cluster variance."""
            overall_mean_risk = np.mean(risks)  # Overall mean risk across all data points
            unique_clusters = np.unique(cluster_labels)
            between_variance = 0

            for cluster in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster)[0]  # Indices of points in this cluster
                cluster_risks = risks[cluster_indices]
                cluster_mean_risk = np.mean(cluster_risks)
                cluster_size = len(cluster_risks)
            
                # Between-cluster variance component
                between_variance += cluster_size * (cluster_mean_risk - overall_mean_risk) ** 2
        
                # Normalize by the number of clusters
                between_variance /= len(unique_clusters)
                return between_variance

        def compute_cpr(between_variance, within_variance):
            """Compute the CPR score as the ratio of between-cluster to within-cluster variance."""
            if within_variance == 0:  # To avoid division by zero
                return float('inf')  # If no within-cluster variance, the clusters are perfectly consistent
            return between_variance / within_variance
        
        # Calculate variances and CPR scores for each clustering method
        # Original LIME clustering
        within_variance_original = calculate_within_cluster_variance(self.data_point_risks_original, self.clusters_original)
        between_variance_original = calculate_between_cluster_variance(self.data_point_risks_original, self.clusters_original)
        cpr_score_original = compute_cpr(between_variance_original, within_variance_original)

        # PCA-transformed LIME clustering
        within_variance_pca = calculate_within_cluster_variance(self.data_point_risks_pca, self.clusters_pca)
        between_variance_pca = calculate_between_cluster_variance(self.data_point_risks_pca, self.clusters_pca)
        cpr_score_pca = compute_cpr(between_variance_pca, within_variance_pca)

        # Clustering with original features
        within_variance_features = calculate_within_cluster_variance(self.data_point_risks_features, self.clusters_features)
        between_variance_features = calculate_between_cluster_variance(self.data_point_risks_features, self.clusters_features)
        cpr_score_features = compute_cpr(between_variance_features, within_variance_features)

        # Store and pretty-print the results
        result = {
        'Original_Clustering': {
            'Within_Cluster_Variance': within_variance_original,
            'Between_Cluster_Variance': between_variance_original,
            'CPR_Score': cpr_score_original
        },
        'PCA_Clustering': {
            'Within_Cluster_Variance': within_variance_pca,
            'Between_Cluster_Variance': between_variance_pca,
            'CPR_Score': cpr_score_pca
        },
        'Feature_Clustering': {
            'Within_Cluster_Variance': within_variance_features,
            'Between_Cluster_Variance': between_variance_features,
            'CPR_Score': cpr_score_features
        }
    }

        # Pretty-print the result
        pprint.pprint(result)

        return result
    
    def load_lime_importances(self, load_path):
        """Load previously saved LIME importances from a file."""
        with open(load_path, 'rb') as file:
            self.lime_importances_df = pickle.load(file)
        print(f"LIME importances loaded from {load_path}")

    def compute_membership_probabilities(self, clustering_method='lime'):
        """
        Compute membership probabilities for KMeans clustering.

        Parameters:
        - clustering_method (str): The method used for clustering ('lime', 'pca', 'features').

        Returns:
        - membership_probs_df: DataFrame containing membership probabilities for the specified clustering method.
        """
        if clustering_method == 'lime':
            kmeans_model = self.kmeans_original
            data_scaled = self.scaler_lime.transform(self.lime_importances_df.drop(columns=['Cluster_Original'], errors='ignore'))

        elif clustering_method == 'pca':
            kmeans_model = self.kmeans_pca
            data_scaled = self.scaler_pca.transform(self.pca_df.drop(columns=['Cluster_PCA'], errors='ignore'))

        elif clustering_method == 'features':
            kmeans_model = self.kmeans_features
            data_scaled = self.scaler_features.transform(self.X_test.drop(columns=['Cluster_Features'], errors='ignore'))

        else:
            raise ValueError("Invalid clustering method. Choose from 'lime', 'pca', or 'features'.")

        # Compute distances to each cluster centroid
        distances = kmeans_model.transform(data_scaled)
        
        # Apply softmax over negative distances to get probabilities
        exp_neg_distances = np.exp(-distances)
        sum_exp_neg_distances = np.sum(exp_neg_distances, axis=1, keepdims=True)
        membership_probs = exp_neg_distances / sum_exp_neg_distances
        
        # Convert to DataFrame
        cluster_labels = [f'Cluster_{i}' for i in range(kmeans_model.n_clusters)]
        membership_probs_df = pd.DataFrame(membership_probs, columns=cluster_labels)
        return membership_probs_df
    
    def add_membership_probabilities_to_dataset(self):
        """
        Add membership probabilities from KMeans clustering (LIME, PCA, and original features) to the test dataset.

        This method modifies self.X_test by adding membership probabilities for each clustering method:
        - 'lime': Membership probabilities based on clustering of LIME importances.
        - 'pca': Membership probabilities based on clustering of PCA-transformed LIME importances.
        - 'features': Membership probabilities based on clustering of the original features.
    
        Returns:
        - data_lime: DataFrame containing the original features plus membership probabilities from 'lime' clustering.
        - data_pca: DataFrame containing the original features plus membership probabilities from 'pca' clustering.
        - data_features: DataFrame containing the original features plus membership probabilities from 'features' clustering.
        """
    
        # Compute membership probabilities for each method
        membership_probs_lime = self.compute_membership_probabilities(clustering_method='lime')
        membership_probs_pca = self.compute_membership_probabilities(clustering_method='pca')
        membership_probs_features = self.compute_membership_probabilities(clustering_method='features')

        # Create new DataFrames with added membership probabilities
        # Reset the index of self.X_test to match the indices of membership probabilities
        self.X_test = self.X_test.reset_index(drop=True)

        # Add LIME-based membership probabilities
        data_lime = pd.concat([self.X_test.copy(), membership_probs_lime], axis=1)
        # Remove Cluster_Features column
        data_lime = data_lime.drop(columns=['Cluster_Features'], errors='ignore')
        # Add T2D column
        data_lime['T2D'] = self.Y
    
        # Add PCA-based membership probabilities
        data_pca = pd.concat([self.X_test.copy(), membership_probs_pca], axis=1)
        # Remove Cluster_Features column
        data_pca = data_pca.drop(columns=['Cluster_Features'], errors='ignore')
        # Add T2D column
        data_pca['T2D'] = self.Y
    
        # Add Feature-based membership probabilities
        data_features = pd.concat([self.X_test.copy(), membership_probs_features], axis=1)
        # Remove Cluster_Features column
        data_features = data_features.drop(columns=['Cluster_Features'], errors='ignore')
        # Add T2D column
        data_features['T2D'] = self.Y

        print("Membership probabilities added to datasets for 'lime', 'pca', and 'features'.")

        # Return the three datasets
        return data_lime, data_pca, data_features
    
    def compute_ari_scores(self, noise_factor=0.1, n_clusters=None):
        """
        Compute the ARI (Adjusted Rand Index) scores for each of the three clustering methods 
        between the original data and the perturbed data.
    
        Parameters:
        - noise_factor: Float, controls the amount of Gaussian noise to add to the data.
        - n_clusters: Integer, the number of clusters for KMeans. If not provided, 
        it will use the number of clusters already calculated.
    
        Returns:
        - ari_scores: Dictionary containing ARI scores for each clustering method (original features, LIME, PCA).
        """
        # Nested function to perturb the data with Gaussian noise
        def perturb_data_with_gaussian_noise(X, noise_factor=0.1):
            """
            Perturb the data with Gaussian noise where the standard deviations
            of the noise are proportional to the feature variances.

            Parameters:
            - X: DataFrame or NumPy array containing the data to perturb.
            - noise_factor: Float, determines the proportion of noise to add, default is 0.1 (10% of the variance).

            Returns:
            - X_perturbed: DataFrame or NumPy array of perturbed data.
            """
            feature_std = X.std(axis=0)  # Calculate the standard deviation for each feature (column)
            noise = np.random.randn(*X.shape) * feature_std.to_numpy() * noise_factor  # Gaussian noise scaled by feature std
            X_perturbed = X + noise  # Add noise to the original data
            return X_perturbed

        # Step 1: Perform clustering on the original test data
        original_X_test = self.X_test.copy()  # Store original test data for later use
        if n_clusters is None:
            n_clusters = len(np.unique(self.clusters_features))  # Default to the original number of clusters
    
        # Train the model, generate LIME importances, perform PCA, and perform clustering
        self.train_model()
        self.generate_lime_importances()
        self.perform_pca()
        self.perform_clustering(n_clusters=n_clusters)
    
        # Store original cluster labels
        original_clusters_features = self.clusters_features.copy()
        original_clusters_lime = self.clusters_original.copy()
        original_clusters_pca = self.clusters_pca.copy()

        # Step 2: Perturb the test data
        X_test_perturbed = perturb_data_with_gaussian_noise(self.X_test[self.X_train.columns], noise_factor=noise_factor)

        # Step 3: Temporarily replace the test data with the perturbed data
        self.X_test = pd.DataFrame(X_test_perturbed, columns=self.X_train.columns, index=self.X_test.index)

        # Step 4: Generate LIME importances, perform PCA and clustering for perturbed data
        self.generate_lime_importances()
        self.perform_pca()
        self.perform_clustering(n_clusters=n_clusters)

        # Step 6: Compute ARI scores between the original and perturbed clusterings
        try:
            # Compare original clusters with clusters from the perturbed data
            ari_score_features = adjusted_rand_score(original_clusters_features, self.clusters_features)
            ari_score_lime = adjusted_rand_score(original_clusters_lime, self.clusters_original)
            ari_score_pca = adjusted_rand_score(original_clusters_pca, self.clusters_pca)
        except KeyError as e:
            print(f"Error computing ARI: {e}")
            raise ValueError("Ensure that the clustering labels are correctly assigned in self.X_test, self.lime_importances_df, and self.pca_df.")

        # Step 7: Restore the original test data
        self.X_test = original_X_test

        # Step 8: Store the ARI scores in a dictionary and return
        ari_scores = {
            'ARI_Features': ari_score_features,
            'ARI_LIME': ari_score_lime,
            'ARI_PCA': ari_score_pca
            }

        # Pretty-print the results
        pprint.pprint(ari_scores)

        return ari_scores
    
    def train_autoencoder(self, latent_dim=6, epochs=50, batch_size=32, verbose=1):
        """
        Train an autoencoder on the training data and store the latent space representation.
        
        Parameters:
        - latent_dim: Integer, the number of dimensions in the latent space (default is 6).
        - epochs: Integer, the number of epochs to train the autoencoder (default is 50).
        - batch_size: Integer, the batch size for training the autoencoder (default is 32).
        - verbose: Integer, the verbosity mode (default is 1, which prints progress bar).
        """
        # Step 1: Standardize the data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Step 2: Define the autoencoder architecture
        input_dim = X_train_scaled.shape[1]
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder part: Compress the input into a lower-dimensional latent space
        encoded = layers.Dense(latent_dim, activation='relu')(input_layer)
        
        # Decoder part: Reconstruct the original input from the latent space
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

        # Create the autoencoder model
        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Step 3: Train the autoencoder
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=epochs, batch_size=batch_size, 
                        shuffle=True, validation_data=(X_test_scaled, X_test_scaled), verbose=verbose)

        # Step 4: Extract the encoder model to get latent space representation
        self.encoder = models.Model(inputs=input_layer, outputs=encoded)

        # Generate the latent space features
        self.latent_train = self.encoder.predict(X_train_scaled)
        self.latent_test = self.encoder.predict(X_test_scaled)
        
        print("Autoencoder trained and latent space features extracted.")

    def perform_autoencoder_clustering(self, n_clusters=7):
        """
        Perform KMeans clustering on the latent space features learned by the autoencoder and 
        order the cluster labels based on T2D risks calculated using the original features.

        Parameters:
        - n_clusters: Integer, the number of clusters for KMeans (default is 7).
        """
        if self.latent_train is None or self.latent_test is None:
            raise ValueError("Autoencoder has not been trained. Please run `train_autoencoder()` first.")

        # Step 1: Perform KMeans clustering on the latent space representation
        self.kmeans_autoencoder = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.clusters_autoencoder_train = self.kmeans_autoencoder.fit_predict(self.latent_train)
        self.clusters_autoencoder_test = self.kmeans_autoencoder.predict(self.latent_test)

        # Store the cluster labels in the latent space DataFrames
        self.latent_train_df = pd.DataFrame(self.latent_train, columns=[f'Latent_{i+1}' for i in range(self.latent_train.shape[1])])
        self.latent_test_df = pd.DataFrame(self.latent_test, columns=[f'Latent_{i+1}' for i in range(self.latent_test.shape[1])])

        self.latent_train_df['Cluster_Autoencoder'] = self.clusters_autoencoder_train
        self.latent_test_df['Cluster_Autoencoder'] = self.clusters_autoencoder_test

        print(f"Autoencoder-based clustering performed with {n_clusters} clusters.")

        # Step 2: Standardize the original features
        scaled_X_test = self.scaler.transform(self.X_test[self.X_train.columns].copy())  # Use only original features
        scaled_X_test_df = pd.DataFrame(scaled_X_test, columns=self.X_train.columns)

        # Step 3: Calculate T2D risks for the autoencoder clusters using original features
        def calculate_risk_for_centroid(scaled_data, cluster_labels):
            scaled_data_df = pd.DataFrame(scaled_data, columns=self.X_train.columns)
        
            # Calculate cluster centroids by grouping the data by cluster labels and taking the mean
            cluster_centroids = scaled_data_df.groupby(cluster_labels).mean()

            # Predict T2D risk using the trained logistic model
            risks = self.logistic_model.predict_proba(cluster_centroids.values)[:, 1]  # Probability of T2D
            return risks

        # Step 4: Compute T2D risks for each cluster in the autoencoder space
        self.risks_autoencoder = calculate_risk_for_centroid(scaled_X_test, self.clusters_autoencoder_test)

        # Step 5: Order the clusters based on T2D risk (ascending)
        ordered_clusters_autoencoder = np.argsort(self.risks_autoencoder)

        # Step 6: Create a mapping from the original cluster labels to the ordered labels
        cluster_risk_order_autoencoder = {i: rank for rank, i in enumerate(ordered_clusters_autoencoder)}

        # Step 7: Update the cluster labels in the latent space DataFrames
        self.latent_train_df['Cluster_Autoencoder_Ordered'] = self.latent_train_df['Cluster_Autoencoder'].map(cluster_risk_order_autoencoder)
        self.latent_test_df['Cluster_Autoencoder_Ordered'] = self.latent_test_df['Cluster_Autoencoder'].map(cluster_risk_order_autoencoder)

        print("Autoencoder-based clusters ordered based on T2D risks.")

    def visualize_autoencoder_feature_association(self, save_path=None):
        """
        Visualize the degree of association between each feature and each autoencoder-based cluster
        using a heatmap where:
        - x-axis: Ordered autoencoder-based cluster labels (based on T2D risk).
        - y-axis: Original feature names.
        - Colors represent the z-score standardized values for each original feature within each cluster.
    
        Parameters:
        - save_path (str, optional): Path to save the output figure. Defaults to None.
        """

        # Step 1: Check if autoencoder clustering is performed and ordered labels are available
        if self.clusters_autoencoder_test is None or 'Cluster_Autoencoder_Ordered' not in self.latent_test_df.columns:
            raise ValueError("Ordered autoencoder-based clusters not available. Please run `perform_autoencoder_clustering()` first.")

        # Step 2: Ensure index alignment before grouping the original features by the ordered autoencoder-based cluster labels
        self.X_test = self.X_test.reset_index(drop=True)
        self.latent_test_df = self.latent_test_df.reset_index(drop=True)

        # Group the original features by the ordered autoencoder-based cluster labels
        X_test_df = pd.DataFrame(self.X_test, columns=self.X_train.columns)  # Restore original features

        # Add the ordered autoencoder cluster labels to the DataFrame
        X_test_df['Cluster_Autoencoder_Ordered'] = self.latent_test_df['Cluster_Autoencoder_Ordered']

        # Step 3: Standardize the original features (z-score)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test_df.drop(columns=['Cluster_Autoencoder_Ordered']))
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=self.X_train.columns)

        # Add the ordered autoencoder cluster labels back to the standardized DataFrame
        X_test_scaled_df['Cluster_Autoencoder_Ordered'] = X_test_df['Cluster_Autoencoder_Ordered']

        # Step 4: Compute the mean of each feature for each ordered cluster
        cluster_means = X_test_scaled_df.groupby('Cluster_Autoencoder_Ordered').mean()

        # Step 5: Create a heatmap of the mean z-scores for each feature in each ordered cluster
        plt.figure(figsize=(5, 3))
        sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5, fmt='.2f')

        # Set labels and title
        plt.xlabel('Ordered Autoencoder-Based Clusters (T2D Risk)')
        plt.ylabel('Original Features')
        plt.title('Feature Association with Ordered Autoencoder-Based Clusters (Z-Score Standardized)')

        # Step 6: Save the figure if a save path is provided
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def generate_shap_values(self, model=None, save_path=None, load_path=None):
        """
        Generate SHAP contribution values for each sample in the test set.
    
        Parameters:
        - model: Trained machine learning model. If None, use the model in the class.
        - save_path (str): Path to save the SHAP values.
        - load_path (str): Path to load previously saved SHAP values.
        """
        if load_path and os.path.exists(load_path):
            with open(load_path, 'rb') as file:
                self.shap_values_df = pickle.load(file)
            print(f"SHAP values loaded from {load_path}")
            return

        # Use the provided model or the one already trained in the class
        model = model or self.model

        # Use TreeExplainer for tree-based models (like RandomForest), and KernelExplainer for other models
        if hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(self.X_train, 100))

        # Compute SHAP values
        shap_values = explainer.shap_values(self.X_test)

        # SHAP for binary classification: use shap_values[1] for the positive class
        shap_values_df = pd.DataFrame(shap_values[1], columns=self.X_train.columns)

        self.shap_values_df = shap_values_df  # Store SHAP values in the class for further use
        print("SHAP values generated and stored.")

        if save_path:
            with open(save_path, 'wb') as file:
                pickle.dump(self.shap_values_df, file)
            print(f"SHAP values saved to {save_path}")

    def perform_shap_clustering(self, n_clusters=7):
        """
        Perform KMeans clustering on the SHAP values and order clusters based on T2D risks.

        Parameters:
        - n_clusters: Number of clusters for KMeans (default is 7).
        """
        if self.shap_values_df is None:
            raise ValueError("SHAP values not available. Please run `generate_shap_values()` first.")
    
        # Scale the SHAP values for clustering
        scaler = StandardScaler()
        scaled_shap_values = scaler.fit_transform(self.shap_values_df)

        # Perform KMeans clustering on the SHAP values
        self.kmeans_shap = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        self.clusters_shap = self.kmeans_shap.fit_predict(scaled_shap_values)
        self.shap_values_df['Cluster_SHAP'] = self.clusters_shap

        print(f"KMeans clustering on SHAP values performed with {n_clusters} clusters.")

        # Standardize the original test features
        scaled_X_test = self.scaler.transform(self.X_test[self.X_train.columns].copy())
        scaled_X_test_df = pd.DataFrame(scaled_X_test, columns=self.X_train.columns)

        # Calculate T2D risks for SHAP clusters using original features
        def calculate_risk_for_centroid(scaled_data, cluster_labels):
            scaled_data_df = pd.DataFrame(scaled_data, columns=self.X_train.columns)
            cluster_centroids = scaled_data_df.groupby(cluster_labels).mean()
            risks = self.logistic_model.predict_proba(cluster_centroids.values)[:, 1]
            return risks

        # Compute T2D risks for each SHAP cluster
        self.risks_shap = calculate_risk_for_centroid(scaled_X_test, self.clusters_shap)

        # Order clusters based on T2D risk (ascending)
        ordered_clusters_shap = np.argsort(self.risks_shap)
        cluster_risk_order_shap = {i: rank for rank, i in enumerate(ordered_clusters_shap)}

        # Update cluster labels in the SHAP values DataFrame
        self.shap_values_df['Cluster_SHAP_Ordered'] = self.shap_values_df['Cluster_SHAP'].map(cluster_risk_order_shap)
        print("SHAP-based clusters ordered based on T2D risks.")

    def visualize_shap_feature_association(self, save_path=None):
        """
        Visualize the degree of association between each feature and each SHAP-based cluster using a heatmap.
    
        - x-axis: SHAP-based cluster labels (ordered by T2D risk).
        - y-axis: Original feature names.
        - Colors represent the z-score standardized values for each original feature within each cluster.
    
        Parameters:
        - save_path (str, optional): Path to save the output figure.
        """
        if 'Cluster_SHAP_Ordered' not in self.shap_values_df.columns:
            raise ValueError("Ordered SHAP-based clusters not available. Please run `perform_shap_clustering()` first.")

        # Ensure index alignment
        self.X_test = self.X_test.reset_index(drop=True)
        self.shap_values_df = self.shap_values_df.reset_index(drop=True)
        
        # Group the original features by the ordered SHAP-based cluster labels
        X_test_df = pd.DataFrame(self.X_test, columns=self.X_train.columns)
        X_test_df['Cluster_SHAP_Ordered'] = self.shap_values_df['Cluster_SHAP_Ordered']

        # Standardize the original features (z-score)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test_df.drop(columns=['Cluster_SHAP_Ordered']))
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=self.X_train.columns)

        # Add the ordered SHAP cluster labels back to the standardized DataFrame
        X_test_scaled_df['Cluster_SHAP_Ordered'] = X_test_df['Cluster_SHAP_Ordered']

        # Compute the mean of each feature for each ordered cluster
        cluster_means = X_test_scaled_df.groupby('Cluster_SHAP_Ordered').mean()

        # Create a heatmap of the mean z-scores for each feature in each ordered cluster
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5, fmt='.2f')

        plt.xlabel('Ordered SHAP-Based Clusters (T2D Risk)')
        plt.ylabel('Original Features')
        plt.title('Feature Association with Ordered SHAP-Based Clusters (Z-Score Standardized)')

        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
    
     # Helper function to add new methods dynamically
    @staticmethod    
    def add_method(cls, func):
        setattr(cls, func.__name__, func)
        print(f"Method {func.__name__} added to {cls.__name__}")