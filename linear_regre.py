import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Load your data
df = pd.read_csv('C:/Users/ADMIN/Desktop/ofc_projects/data_pipeline/other_datasets/Admission_Predict.csv')

# Rename columns (if needed)
df.columns = df.columns.str.replace(' ', '_').str.rstrip('_')

print(df.dtypes)

print(df.shape)

print(df.info())

print(df.describe())

print(df.loc[:,df.any()])


# Create and fit the linear regression model
model = LinearRegression()
X = df[['GRE_Score']]  # Independent variable (features)
y = df['Chance_of_Admit']  # Dependent variable (target)

model.fit(X, y)

# Accept user input for GRE_Score
user_gre_score = int(input("Enter your GRE Score: "))
# Make a prediction for the user input

predicted_chance_of_admit = model.predict([[user_gre_score]])

# Display the prediction
print(f'Predicted Chance of Admission: {predicted_chance_of_admit[0] * 100:.2f}%')

# # Calculate R-squared to evaluate the performance of your regression model
# r_squared = r2_score(y, model.predict(X))
# print(f'R-squared (R²): {r_squared:.4f}')


# # # Make predictions
# # predicted_chance_of_admit = model.predict(X)

# Visualize the results
plt.scatter(X, y, color='red', marker='+', label='Data Points')
plt.xlabel('GRE_Score')
plt.ylabel('Chance_of_Admit')
plt.plot(X, model.predict(X), color='blue', label='Linear Regression Line')
plt.legend()
plt.title('Linear Regression: GRE_Score vs Chance_of_Admit')
plt.show()
print(model.coef_)
print(model.intercept_)
accury=model.score(df[['GRE_Score']],df['Chance_of_Admit'])
print(accury*100,'%')

print()














# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model

# df = pd.read_csv('C:/Users/ADMIN/Desktop/data_pipeline/Admission_Predict.csv')

# # Rename the columns by replacing spaces with underscores and removing trailing underscores
# df.columns = df.columns.str.replace(' ', '_').str.rstrip('_')

# # Scatter plot
# plt.scatter(df['GRE_Score'], df['Chance_of_Admit'], color='red', marker='+')
# plt.xlabel('GRE_Score')
# plt.ylabel('Chance_of_Admit')
# plt.show()

# # Create and fit the linear regression model
# model = linear_model.LinearRegression()
# X = df[['GRE_Score']]  # Features (2D array)
# y = df['Chance_of_Admit']  # Target (1D array)

# model.fit(X, y)

# # Predict for a specific GRE score (e.g., 230)
# predicted_admit_chance = model.predict([[230]])
# print(f'Predicted Chance of Admission for GRE Score 230: {predicted_admit_chance[0]}')

















# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model

# df=pd.read_csv('C:/Users/ADMIN/Desktop/data_pipeline/Admission_Predict.csv')

# # Rename the columns by replacing spaces with underscores
# df.columns = df.columns.str.replace(' ', '_')
# # Remove underscores at the end of column names
# df.columns = df.columns.str.rstrip('_')
# print(df.columns)
# #df_predict=df[['GRE_Score','Chance_of_Admit']]
# #print(df_predict.head())
# plt.scatter(df['GRE_Score'],df['Chance_of_Admit'],color='red',marker='+')
# plt.xlabel('GRE_Score')
# plt.ylabel('Chance_of_Admit')
# plt.show()
# model=linear_model.LinearRegression()
# model.fit(df['GRE_Score'],df['Chance_of_Admit'])
# model.predict([230])
