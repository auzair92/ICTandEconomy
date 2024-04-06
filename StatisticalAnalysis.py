import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = 'data/time_series_data_year_cv.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe and its summary information
data_head = data.head()
data_info = data.info()

data_head, data_info

# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Create a figure and a set of subplots for visualizing the distributions of the variables
fig, axs = plt.subplots(4, 3, figsize=(20, 20))

# Flatten the axes array for easy indexing
axs = axs.flatten()

# List of independent variables
independent_vars = ['MBB_Per_100', 'FBB_Per_100', 'Internet_Users_Perc', 'Mob_Cellular_Sub', 'ICT_Services_Exp', 'Infl_CP_Annual_Perc', 'Unempl_Perc_LF', 'Population', 'Manuf_Val_Add', 'Exp_Good_Serv', 'Gross_Cap_Form']

# Plotting histograms for each independent variable
for i, var in enumerate(independent_vars):
    sns.histplot(data[var], kde=True, ax=axs[i], color='skyblue')
    axs[i].set_title(f'Distribution of {var}', fontsize=15)
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')

# Adjusting the layout
plt.tight_layout()

# Displaying the plot for the dependent variable separately for clarity
plt.figure(figsize=(8, 6))
sns.histplot(data['GDP_Per_Capita'], kde=True, color='salmon')
plt.title('Distribution of GDP_Per_Capita', fontsize=15)
plt.xlabel('')
plt.ylabel('')
plt.show()

# Calculate Pearson's correlation coefficient
pearson_corr = data[independent_vars + ['GDP_Per_Capita']].corr(method='pearson')

# Calculate Spearman's rank correlation
spearman_corr = data[independent_vars + ['GDP_Per_Capita']].corr(method='spearman')

# Visualization of Pearson's correlation coefficients
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Coefficients')
plt.show()

# Visualization of Spearman's rank correlation coefficients
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Rank Correlation Coefficients')
plt.show()

pearson_corr['GDP_Per_Capita'], spearman_corr['GDP_Per_Capita']

# Log transformation of independent variables and the dependent variable, excluding 'Year' and 'Country'
data_transformed = data.copy()
for var in independent_vars + ['GDP_Per_Capita']:
    # Adding a small constant to avoid log(0) where necessary
    data_transformed[var] = np.log(data_transformed[var] + 1)

# Splitting the dataset into training and testing sets
X = data_transformed[independent_vars]
y = data_transformed['GDP_Per_Capita']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculating R^2 and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculating VIF for each independent variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Display the r2, rmse, vif values
r2, rmse, vif_data

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=0.95) # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Splitting the PCA-transformed data into training and testing sets
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Multiple Linear Regression using PCA components
model_pca = LinearRegression()
model_pca.fit(X_pca_train, y_train)

# Predictions
y_pca_pred = model_pca.predict(X_pca_test)

# Calculating R^2 and RMSE for the PCA-based model
r2_pca = r2_score(y_test, y_pca_pred)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pca_pred))

# Since each PCA component is orthogonal (independent) to each other, VIF should theoretically be 1 for all components
# However, calculating VIF for demonstration purposes
vif_data_pca = pd.DataFrame()
vif_data_pca["PCA Component"] = range(1, X_pca_train.shape[1] + 1)
vif_data_pca["VIF"] = [variance_inflation_factor(X_pca_train, i) for i in range(X_pca_train.shape[1])]

r2_pca, rmse_pca, vif_data_pca, pca.n_components_

# Retrieving the explained variance ratio of the principal components
explained_variance_ratio = pca.explained_variance_ratio_

# Creating a dataframe to display the explained variance by each principal component
explained_variance_df = pd.DataFrame(explained_variance_ratio, columns=['Explained Variance Ratio'], index=[f'PC{i+1}' for i in range(len(explained_variance_ratio))])
explained_variance_df['Cumulative Explained Variance'] = explained_variance_df['Explained Variance Ratio'].cumsum()

explained_variance_df

# Plotting the explained variance ratio and cumulative explained variance
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio', color=color)

ax1.bar(explained_variance_df.index, explained_variance_df['Explained Variance Ratio'], color=color, alpha=0.6, label='Explained Variance Ratio')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Explained Variance', color=color)
ax2.plot(explained_variance_df.index, explained_variance_df['Cumulative Explained Variance'], color=color, marker='o', label='Cumulative Explained Variance')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')

fig.tight_layout()
plt.title('PCA Explained Variance Ratio and Cumulative Explained Variance')
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.show()

# Getting the PCA component loadings (the coefficients of the original variables in each component)
component_loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Displaying the component loadings
component_loadings

# Calculating the correlation matrix for the principal component loadings
pc_correlation_matrix = component_loadings.corr(method='pearson')

# Plotting the heatmap (optional)
plt.figure(figsize=(10, 8))
sns.heatmap(pc_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap of Principal Component Loadings')
plt.show()
