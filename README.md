Topic Proposal 

 
 

Team Members: QiXiong(Cody) Yu, Tracie Robinson, Xuyue Wan, Indulekha Siddabathuni 

 
 

Topic: Banking Loan Prediction 

 
 

A bank’s digital division is encountering obstacles in converting leads into customers.  Their main goal is to enhance customer acquisition through digital means.  The main focus is to increase the number of leads that enter the conversion.  They obtain leads through search engines, display ads, email campaigns, and through affiliate partners.  Naturally, the conversion rate varies depending on the source and quality of the leads. 

The current objective is to identify the segments of leads that have higher conversion rate (from lead to purchasing a product).  This will enable them to specifically target these potential customers through additional channels and re-marketing strategies.  The data set contains a customers’ loan history from the past three months, as well as their basic information. 

 Our data set is from the Kaggle, which contains 30,038 rows and 22 columns. 

 
 

Attributes:  

ID : Unique Customer ID 

Gender : Gender of the applicant 

DOB : Date of Birth of the applicant 

Lead_Creation_Date : Date on which Lead was created 

City_Code : Anonymised Code for the City 

City_Category: Anonymised City Feature 

Employer_Code: Anonymised Code for the Employer 

Employer_Category1 : Anonymised Employer Feature 

Employer_Category2: Anonymised Employer Feature 

Monthly_Income : Monthly Income in Dollars 

Customer_Existing_Primary_Bank_Code : Anonymised Customer Bank Code 

Primary_Bank_Type: Anonymised Bank Feature 

Contacted: Contact Verified (Y/N) 

Source : Categorical Variable representing source of lead 

Source_Category: Type of Source 

Existing_EMI : EMI of Existing Loans in Dollars 

Loan_Amount: Loan Amount Requested 

Loan_Period: Loan Period (Years) 

Interest_Rate: Interest Rate of Submitted Loan Amount 

EMI: EMI of Requested Loan Amount in dollars 

Var1: Anonymized Categorical variable with multiple levels 

Approved: (Target) Whether a loan is Approved or not (1-0) . Customer is Qualified Lead or not (1-0) 

SMART questions: 

What is the proportion of approved and rejected loan applications in the dataset? 

Can we identify a set of key features that are most predictive of whether or not an applicant will be approved or denied?  

Can we do dimensionality reduction based on anonymized features such as city and employer? 

Are there any variables that indicate how much an applicant will be approved for? 

Is there any correlation between the monthly income and the interest rate for the applicant’s loan? What about loan amount and interest rate? Loan period and interest rate? 

 
 

Models: Exploratory Data Analysis (EDA),  Simple Linear Regression(SLR), Multi-Linear Regression(MLR), T-test, Logistic regression, Decision tree or Random forest 

  

Dataset Source: Kaggle dataset : Banking Loan Prediction 

Link: https://www.kaggle.com/datasets/arashnic/banking-loan-prediction  

  

GitHub Repo Link: https://github.com/XuyueW/Data-Mining-Project   

 
Instructor's feedback: 

This is very well-written proposal. Crisp and clear! 

You are encouraged to also consider some other machine learning model. 

Please make sure that you define your target variable clearly in the report. I see that there are multiple outcomes you are trying to predict, such as approval status, approved amount (for those who got approved), segments of leads with high conversion rate... 

 