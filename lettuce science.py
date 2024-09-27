import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Load the datasets
lettuce_df = pd.read_csv('/kaggle/input/lettuce-growth-days/lettuce_dataset.csv', encoding='ISO-8859-1')
unseen_df = pd.read_csv('/kaggle/input/lettuce-growth-days/unseen_data.csv', encoding='ISO-8859-1')
lettuce_updated_df = pd.read_csv('/kaggle/input/lettuce-growth-days/lettuce_dataset_updated.csv', encoding='ISO-8859-1')

#data analysis
lettuce_df.head(), unseen_df.head(), lettuce_updated_df.head()
lettuce_df['Date'] = pd.to_datetime(lettuce_df['Date'])
unseen_df['Date'] = pd.to_datetime(unseen_df['Date'])
lettuce_updated_df['Date'] = pd.to_datetime(lettuce_updated_df['Date'])



lettuce_df.describe()
# Correlation heatmap
numeric_df = lettuce_df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(x='Temperature (°C)', y='Growth Days', data=lettuce_df, ax=axes[0, 0])
sns.scatterplot(x='Humidity (%)', y='Growth Days', data=lettuce_df, ax=axes[0, 1])
sns.scatterplot(x='TDS Value (ppm)', y='Growth Days', data=lettuce_df, ax=axes[1, 0])
sns.scatterplot(x='pH Level', y='Growth Days', data=lettuce_df, ax=axes[1, 1])
plt.tight_layout()
plt.show()


# Prepare the data for modeling
X = lettuce_df[['Temperature (°C)', 'Humidity (%)', 'TDS Value (ppm)', 'pH Level']]
y = lettuce_df['Growth Days']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse, r2