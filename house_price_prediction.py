# house_price_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("train.csv")
print("✅ Dataset Loaded:", df.shape)

# Step 2: Select Features
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Step 3: Visualize Feature Relationships (using matplotlib)
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(df['GrLivArea'], df['SalePrice'], color='blue', alpha=0.5)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('GrLivArea vs SalePrice')

plt.subplot(2, 2, 2)
plt.scatter(df['BedroomAbvGr'], df['SalePrice'], color='green', alpha=0.5)
plt.xlabel('BedroomAbvGr')
plt.ylabel('SalePrice')
plt.title('Bedrooms vs SalePrice')

plt.subplot(2, 2, 3)
plt.scatter(df['FullBath'], df['SalePrice'], color='orange', alpha=0.5)
plt.xlabel('FullBath')
plt.ylabel('SalePrice')
plt.title('Bathrooms vs SalePrice')

plt.tight_layout()
plt.savefig("pairplot.png")
plt.show()

# Step 4: Prepare Data
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Step 5: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model Trained")

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Squared Error: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
