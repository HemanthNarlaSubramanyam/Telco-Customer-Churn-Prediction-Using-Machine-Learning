#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, ConfusionMatrixDisplay, precision_score, recall_score


# In[2]:


df = pd.read_csv("/Users/hemanthnarlasubramanyam/Documents/Projects/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn 2.csv" , index_col = 'customerID')

df.head()


# In[3]:


df.shape


# In[4]:


df_columns = df.columns.tolist()
for column in df_columns:
    print(f"{column} unique values : {df[column].unique()}")


# In[5]:


### Exploratory Data Analysis (EDA)


# In[6]:


# Statistic descriptive
df.describe()


# In[7]:


### We can take a conclusion from that is :

- SeniorCitizen must be categorical data because has minimum value is 0 and the maximum value is 1.
- The average customer stayed in the company is 32 months and 75% of customer has a tenure of 55 month
- Average monthly charges are USD 64.76 and 25% of customers pay more than USD 89.85


# In[8]:


# Change TotalCharges to float
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[9]:


### Payment Method Check


# In[10]:


df["PaymentMethod"].unique()


# In[13]:


# Delete "automatic" from PaymentMethod
df["PaymentMethod"] = df["PaymentMethod"].str.replace(" (automatic)", "", regex=False)


# In[14]:


### Missing Values


# In[15]:


features_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]

for feature in features_na:
    print(f"{feature}, {round(df[feature].isnull().mean(), 4)} % Missing values")


# In[16]:


# Check observation of missing values
df[df[features_na[0]].isnull()]


# In[17]:


# Drop missing values
df.dropna(inplace=True)


# In[18]:


### Target Variable Visualization


# In[19]:


#Apply the ggplot style
plt.style.use("ggplot")


# In[20]:


plt.figure(figsize=(5,5))
ax = sns.countplot(x = df["Churn"],palette="Blues")
ax.bar_label(ax.containers[0])
plt.show()


# The following bar plot shows the target variable of churn yes and no. The proportion of churn is an imbalanced data set because both classes are not equally distributed. To handle it, resampling would be a suitable approach. To keep this simple, we will keep the imbalanced data set and uses many evaluation matrices to evaluate models.

# ### Analysis Services Each Customer

# In[21]:


#Make a function to plot categorical data according to target
def plot_categorical_to_target(df,categorical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(categorical_values)/2)
    
    fig = plt.figure(figsize = (12, 5*number_of_rows))
    
    for index, column in enumerate(categorical_values, 1):
        ax = fig.add_subplot(number_of_rows,number_of_columns,index)
        ax = sns.countplot(x = column, data = df, hue = target, palette="Blues")
        ax.set_title(column)
    return plt.show()


# In[22]:


customer_services = ["PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
plot_categorical_to_target(df,customer_services, "Churn")


# ### We can extract the following conclusions by evaluating service attributes
# 
# - The moderately higher churn rate for customers who has the phone service.
# - Customers with internet service fiber optic have a higher churn rate compared with DSL and No.
# - The much higher churn rate for customers without online security.
# - Customers who don’t have access to tech support tend to leave more frequently than those who do.
# - Customers without online backup and device protection have a higher churn rate.

# ### Analysis Customer Account Information — Categorical Variables

# In[23]:


customer_account_cat = ["Contract","PaperlessBilling","PaymentMethod"]
plot_categorical_to_target(df,customer_account_cat,"Churn")


# ### The following bar plot shown above can make conclusions from that:
# 
# - Customers are more likely to churn with month-to-month contracts.
# - Moderately higher churn rate with electronic check payment method.
# - Customers with paperless billing have higher churn rates.

# # Analysis Customer Account Information — Numerical Variables

# In[24]:


def histogram_plots(df, numerical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values)/2)
    
    fig = plt.figure(figsize=(12,5*number_of_rows))
    
    for index, column in enumerate(numerical_values,1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.kdeplot(df[column][df[target]=="Yes"] ,fill = True)
        ax = sns.kdeplot(df[column][df[target]=="No"], fill = True)
        ax.set_title(column)
        ax.legend(["Churn","No Churn"], loc='upper right')
    plt.savefig("numerical_variables.png", dpi=300)
    return plt.show()


# In[25]:


customer_account_num = ["tenure", "MonthlyCharges","TotalCharges"]
histogram_plots(df,customer_account_num, "Churn")


# ### The following histograms above we can get conclusions that:
# 
# - Customers with short tenure are more churn.
# - Customers with paid more on monthly charges have higher churn rates.
# - Customers with high total charges tend to churn.

# # Analysis of Customer's Demographic Info

# In[26]:


customer_demo_info = ["gender","SeniorCitizen","Partner","Dependents"]
plot_categorical_to_target(df, customer_demo_info,"Churn")


# ### The following bar plot above we can draw some conclusions:
# 
# - Churn and no churn no have differences for each gender.
# - Young customers are more likely to churn rather than old customers.
# - Customers with a partner are less than churn if compared with a partner.

# ## Check Outliers Using Boxplot

# In[27]:


def outlier_check_boxplot(df,numerical_values):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values)/2)
    
    fig = plt.figure(figsize=(12,5*number_of_rows))
    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.boxplot(x = column, data = df, palette = "Blues")
        ax.set_title(column)
    plt.savefig("Outliers_check.png", dpi=300)
    return plt.show()


# In[28]:


numerical_values = ["tenure","MonthlyCharges","TotalCharges"]
outlier_check_boxplot(df,numerical_values)


# ### From boxplots, we can take a conclusion that each numerical variable doesn’t have an outlier.

# # Feature Engineering

# In[30]:


feature_le = ["Partner","Dependents","PhoneService", "Churn","PaperlessBilling"]
def label_encoding(df,features):
    for i in features:
        df[i] = df[i].map({"Yes":1, "No":0})
    return df

df = label_encoding(df,feature_le)
df["gender"] = df["gender"].map({"Female":1, "Male":0})


# ## One Hot Encoding

# In[31]:


features_ohe = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
df_ohe = pd.get_dummies(df, columns=features_ohe)


# ## Feature Scaling

# In[32]:


features_mms = ["tenure","MonthlyCharges","TotalCharges"]

df_mms = pd.DataFrame(df_ohe, columns=features_mms)
df_remaining = df_ohe.drop(columns=features_mms)

mms = MinMaxScaler(feature_range=(0,1))
rescaled_feature = mms.fit_transform(df_mms)

rescaled_feature_df = pd.DataFrame(rescaled_feature, columns=features_mms, index=df_remaining.index)
df = pd.concat([rescaled_feature_df,df_remaining],axis=1)


# ## Correlation Analysis

# In[34]:


plt.figure(figsize=(10,6))
df.corr()["Churn"].sort_values(ascending=False).plot(kind="bar")
plt.savefig("correlation.png", dpi=300)
plt.show()


# In[35]:


X = df.drop(columns = "Churn")
y = df.Churn

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Creates Function For Displaying Evaluation Metrics

# In[36]:


# For logistic Regression
def feature_weights(X_df, classifier, classifier_name):
    weights = pd.Series(classifier.coef_[0], index = X_df.columns.values).sort_values(ascending=False)
    
    top_10_weights = weights[:10]
    plt.figure(figsize=(7,6))
    plt.title(f"{classifier_name} - Top 10 Features")
    top_10_weights.plot(kind="bar")
    
    bottom_10_weights = weights[len(weights)-10:]
    plt.figure(figsize=(7,6))
    plt.title(f"{classifier_name} - Bottom 10 Features")
    bottom_10_weights.plot(kind="bar")
    print("")


# In[37]:


def confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred, classifier, classifier_name):
    cm = confusion_matrix(y_pred,y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.show()
    
    print(f"Accuracy Score Test = {accuracy_score(y_pred,y_test)}")
    print(f"Accuracy Score Train = {classifier.score(X_train,y_train)}")
    return print("\n")


# In[38]:


def roc_curve_auc_score(X_test, y_test, y_pred_probabilities,classifier_name):
    y_pred_prob = y_pred_probabilities[:,1]
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    
    plt.plot([0,1],[0,1],"k--")
    plt.plot(fpr,tpr,label=f"{classifier_name}")
    plt.title(f"{classifier_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return print(f"AUC Score (ROC):{roc_auc_score(y_test,y_pred_prob)}")


# In[39]:


def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
    y_pred_prob = y_pred_probabilities[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall,precision, label=f"{classifier_name}")
    plt.title(f"{classifier_name}-ROC Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    f1_score_result, auc_score = f1_score(y_test,y_pred), auc(recall,precision)
    return print(f"f1 Score : {f1_score_result} \n AUC Score (PR) : {auc_score}")


# # K-Nearest Neighbor

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_proba = knn.predict_proba(X_test)


# In[41]:


confusion_matrix_plot(X_train,y_train,X_test, y_test, y_pred_knn, knn, "K-Nearest Neighbors")


# In[42]:


roc_curve_auc_score(X_test,y_test,y_pred_knn_proba, "K-Nearest Neighbors")


# In[43]:


precision_recall_curve_and_scores(X_test,y_test,y_pred_knn,y_pred_knn_proba,"K-Nearest Neighbors")


# ## Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_proba = logreg.predict_proba(X_test)


# In[45]:


feature_weights(X_train,logreg,"Logistic Regression")


# In[46]:


confusion_matrix_plot(X_train,y_train,X_test,y_test, y_pred_logreg,logreg,"Logistic Regression")


# In[48]:


roc_curve_auc_score(X_test,y_test,y_pred_logreg_proba, "Logistic Regression")


# In[49]:


precision_recall_curve_and_scores(X_test,y_test,y_pred_knn,y_pred_logreg_proba,"Logistic Regression")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




