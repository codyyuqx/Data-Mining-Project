#%%
# EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('train.csv')
df.head()
# drop ID
df = df.drop('ID', axis=1)
print("Shape of dataset:", df.shape)
print("Data types:\n", df.dtypes)
# check missing data
print("Number of missing values:\n", df.isna().sum())
# for small number of missing data, we can try fill it or drop those observations.
# for large number of categorical missing data, we can consider grouping the categories with missing values into a new category.
# for large number of numeric missing data, we might try to delete that variable.
# convert Employer_Category2 into categorical
df['Employer_Category2'] = df['Employer_Category2'].astype(str)
# convert Var1 (Var1: Anonymized Categorical variable with multiple levels)
df['Var1'] = df['Var1'].astype(str)
# convert Approved (Whether a loan is Approved or not (1-0) . Customer is Qualified Lead or not (1-0))
df['Approved'] = df['Approved'].astype(str)
print("Data types:\n", df.dtypes)
# numeric variable summary
df.describe()
# df['Employer_Code'].unique()
# df['Customer_Existing_Primary_Bank_Code'].unique()
# df['Employer_Code'].unique()

#%%
# missing value and convert data
# for 15 missing value in DOB, we can just drop those observations
df = df.dropna(subset=['DOB'])
# convert DOB to age in years

now = datetime.now()
df['age'] = now.year - pd.to_datetime(df['DOB'], format='%d/%m/%y').dt.year
# drop DOB column
df = df.drop('DOB', axis=1)

# We find age have unreal data including birth in the future, drop them
print(f"Number of positive values: {(df['age'] >= 0).sum()}")
print(f"Number of negative values: {(df['age'] < 0).sum()}")
df = df[df['age'] >= 0] 

# also convert and check Lead_Creation_Date 
df['lead_years'] = now.year - pd.to_datetime(df['Lead_Creation_Date'], format='%y/%m/%d').dt.year
df = df.drop('Lead_Creation_Date', axis=1)
print(df['lead_years'].unique())
print(f"Number of positive values: {(df['lead_years'] >= 0).sum()}")
print(f"Number of negative values: {(df['lead_years'] < 0).sum()}")
# there are 17044 negative values, we should drop them
df = df[df['lead_years'] >= 0] 
# now age and lead_years should be numeric

# Find the most frequent category in City_Code
most_frequent = df['City_Code'].mode()[0]
num_occurrences = (df['City_Code'] == most_frequent).sum()
print(f"The most frequent category in City_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# it is not a good idea to fill with mode as number of mode is 7250
# drop missing values
df = df.dropna(subset=['City_Code'])
# missing value in City_Category was dropped simultaneously 

# Find the most frequent category in Employer_Code 
most_frequent = df['Employer_Code'].mode()[0]
num_occurrences = (df['Employer_Code'] == most_frequent).sum()
print(f"The most frequent category in Employer_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# if we fill missing value with mode, the distribution will change, better to drop them 
# df['Employer_Code'] = df['Employer_Code'].fillna(most_frequent)
df = df.dropna(subset=['Employer_Code'])
# # missing value in Employer_Category1 was dropped simultaneously 

# Find the most frequent category in Customer_Existing_Primary_Bank_Code
most_frequent = df['Customer_Existing_Primary_Bank_Code'].mode()[0]
num_occurrences = (df['Customer_Existing_Primary_Bank_Code'] == most_frequent).sum()
print(f"The most frequent category in Customer_Existing_Primary_Bank_Code is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Customer_Existing_Primary_Bank_Code'])
# missing value in Primary_Bank_Type was dropped simultaneously 

# Find the most frequent category in Loan_Amount
most_frequent = df['Loan_Amount'].mode()[0]
num_occurrences = (df['Loan_Amount'] == most_frequent).sum()
print(f"The most frequent category in Loan_Amount is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Loan_Amount'])
# missing value in Loan_Period was dropped simultaneously

# Find the most frequent category in Interest_Rate 
most_frequent = df['Interest_Rate'].mode()[0]
num_occurrences = (df['Interest_Rate'] == most_frequent).sum()
print(f"The most frequent category in Interest_Rate is {most_frequent}")
print(f"It occurs {num_occurrences} times")
# drop missing values
df = df.dropna(subset=['Interest_Rate'])
# missing value in EMI was dropped simultaneously

# Contacted only left Y,drop it
print(f"Number of positive values: {(df['Contacted'] == 'Y').sum()}")
df = df.drop('Contacted', axis=1)

# check again 
print("Number of missing values:\n", df.isna().sum())
print("Shape of dataset:", df.shape)
# Shape of dataset: (14410, 20)

#%% 
# dimensional reduction  
# (City_Code , City_Category) 
# (Employer_Code,Employer_Category1,Employer_Category2)
# (Customer_Existing_Primary_Bank_Code,Primary_Bank_Type)
# (Source,Source_Category)

# Perform chi-square test of independence between Employer_Category1,Employer_Category2

observed = pd.crosstab(df['Employer_Category1'], df['Employer_Category2'])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.2f}")

# Combine two highly correlated categorical variables using PCA


# (City_Code , City_Category) 
X = pd.get_dummies(df[['City_Code', 'City_Category']])
pca = PCA(n_components=1)
df['City_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['City_Code', 'City_Category'])

# (Employer_Code,Employer_Category1,Employer_Category2)(will run for a few minute)
X = pd.get_dummies(df[['Employer_Code','Employer_Category1','Employer_Category2']])
pca = PCA(n_components=1)
df['Employer_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Employer_Code','Employer_Category1','Employer_Category2'])

# (Customer_Existing_Primary_Bank_Code,Primary_Bank_Type)
X = pd.get_dummies(df[['Customer_Existing_Primary_Bank_Code','Primary_Bank_Type']])
pca = PCA(n_components=1)
df['Bank_Type_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Customer_Existing_Primary_Bank_Code','Primary_Bank_Type'])

# check independence of (Source,Source_Category)
observed = pd.crosstab(df['Source'], df['Source_Category'])
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p:.2f}")

# (Source,Source_Category)
X = pd.get_dummies(df[['Source','Source_Category']])
pca = PCA(n_components=1)
df['Source_Combined'] = pca.fit_transform(X)
# drop original variables
df = df.drop(columns=['Source','Source_Category'])

# %%
# bar plot of Approved
plt.figure(figsize=(6,4))
plt.bar(df['Approved'].unique(), df['Approved'].value_counts())
plt.title('Distribution of Approved Status') 
plt.xlabel('Approved Status')
plt.ylabel('Count')
plt.show()
# extreme imbalance data (14060 vs 350), will affect the performance of machine learning
print(df['Approved'].value_counts())
# Solution 1: train this dataset however may have a poor prediction performance on the minority class
# Solution 2: Generate synthetic samples: You can use techniques like Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples of the minority class.

# correlation matrix for numeric variables
numerical_vars = ['Monthly_Income', 'Existing_EMI', 'Loan_Amount', 'Loan_Period','Interest_Rate','EMI']
corr_matrix = df[numerical_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
# EMI vs loan amount: 0.91
# interest rate vs loan amount: -0.3
# loan preiod vs loan amount: 0.37
# existing EMI vs monthly income: 0.17
# EMI vs interest rate: -0.23 

#%%
# SMOTE(generate synthetic samples)


# note SMOTE requires all variable involved are numeric
# categorical_cols = df.iloc[:, [0, 7, 8]]
# df_dummy = pd.get_dummies(df, columns=categorical_cols)

# in case of confusion, create a copy of dataset 
df_dummy = df.copy()
df_dummy['Gender'] = df_dummy['Gender'].map({'Female': 0, 'Male': 1})
df_dummy['Approved'] = df_dummy['Approved'].astype(int)
df_dummy['Var1'] = df_dummy['Var1'].astype(int)

# Apply SMOTE to balance binary response variable
smote = SMOTE()
X = df_dummy.drop('Approved', axis=1)
y = df_dummy['Approved']
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the first few rows of the resampled data
resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
resampled_data.head()
print("Shape of balanced dataset:", resampled_data.shape)
# Shape of balanced dataset: (28120, 15)
# We have a balanced dataset for machine learning
print(resampled_data['Approved'].value_counts())

#%%
# Prepare the data for modeling
X_loan_amount = df['Loan_Amount'].values.reshape(-1, 1)
y_EMI = df['EMI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_loan_amount,y_EMI, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict on the testing set
y_pred = lr_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy:', accuracy)

#sns.swarmplot(x='EMI', y='Monthly_Income', data=df)

#sns.violinplot(x='EMI', y='Monthly_Income', data=df)
sns.boxplot(x='EMI', y='Loan_Amount', data=df)

# %%
# Split data into training and test sets
X_amount_EMI = df[['Loan_Amount', 'EMI']]
y_approved= df['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the KNN model
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")