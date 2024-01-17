#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt

import warnings
warnings.filterwarnings("ignore")

# https://data.worldbank.org/indicator/EN.ATM.CO2E.KT


# In[2]:


def worldbank_data(filename):
    """
    Reads World Bank data from a CSV file.

    Input:
    - filename: The path to the CSV file containing World Bank data.

    Output:
    - years_df: Transposed DataFrame with 'Country Name' as the index.
    - worldbank_df: Original DataFrame read from the CSV file.
    """
    worldbank_df = pd.read_csv(filename, skiprows=4)
    years_df = worldbank_df.set_index(['Country Name']).T
    return years_df, worldbank_df


# In[3]:


years_df, countries_df = worldbank_data('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_6299644.csv')


# In[4]:


countries_df.head()


# In[5]:


def select_indicator_data(countries_df, start_year, end_year):
    """
    Selects indicator data for the specified years from the given DataFrame.

    Returns:
    - DataFrame: Subset of the input DataFrame containing 'Country Name', 'Indicator Name', and the specified years' data.
    """
    selected_columns = ['Country Name', 'Indicator Name'] + list(map(str, range(start_year, end_year + 1)))
    df = countries_df[selected_columns]

    return df

data = select_indicator_data(countries_df, 2010, 2020)


# In[6]:


data.head()


# In[7]:


data = data.dropna()


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


Growth = data[["Country Name", "2020"]].copy()
Growth['2020'] = Growth['2020'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
Growth = Growth.assign(Growth_Percentage=lambda x: 100.0 * (data["2020"] - data["2010"]) / data["2010"])
Growth.head()


# In[11]:


Growth.describe()


# In[12]:


plt.figure(figsize=(8, 8))
sns.scatterplot(x=Growth["2020"], y=Growth["Growth_Percentage"], label="CO2 Emissions Data")
plt.xlabel("CO2 Emissions in 2020")
plt.ylabel("Percentage Growth from 2000 to 2021")
plt.title("Scatter Plot of CO2 Emissions in 2020 vs. Percentage Growth (2000 to 2021)")
plt.show()


# In[13]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[14]:


scaler = StandardScaler()
Growth2 = Growth[["2020", "Growth_Percentage"]]
scaler.fit(Growth2)
Growth_norm = scaler.transform(Growth2)

silhouette_scores = []
for i in range(2, 11):
    score = one_silhouette(Growth_norm, i)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
optimal_clusters = np.argmax(silhouette_scores) + 3 
plt.axvline(x=optimal_clusters, linestyle='--', color='black', label=f'Optimal Clusters (k={optimal_clusters})')
plt.legend()
plt.show()


# In[15]:


kmeans = KMeans(n_clusters=4, n_init=20)
kmeans.fit(Growth_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(Growth["2020"], Growth["Growth_Percentage"], 10, labels, marker="o")
plt.scatter(xkmeans, ykmeans, 50, "k", marker="d")
plt.xlabel("CO2 Emissions in 2020")
plt.ylabel("Percentage Growth from 2010 to 2020")
plt.title("K-Means Clustering of CO2 Emissions")
plt.show()


# In[16]:


years_df.head()


# In[17]:


World_data = years_df[['World']]
World_data = World_data.loc['2010':'2020']
World_data.reset_index(inplace=True)
World_data.columns.name = 'Index'
World_data.rename(columns={'index': 'Years'}, inplace=True)
World_data


# In[18]:


World_data['World'] = World_data['World'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
World_data['Years'] = World_data['Years'].apply(lambda x: pd.to_numeric(x, errors='coerce'))


# In[19]:


World_data.describe()


# In[20]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=World_data, x='Years', y='World')
plt.xlabel('Years')
plt.ylabel('CO2 Emissions (kt)')
plt.title('CO2 Emissions Over Time (2010-2020)')
plt.show()


# In[21]:


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    # makes it easier to get a guess for initial parameters
    t = t - 2010
    f = n0 * np.exp(g*t)
    return f


# In[22]:


param, covar = opt.curve_fit(exponential, World_data["Years"], World_data["World"], p0=(1.1e7, 0.2))
print(f"CO2 Emissions in 2010: {param[0]/1e6:6.1f} million kt")
print(f"Growth Rate: {param[1]*100:4.2f}%")


# In[25]:


World_data["trial"] = exponential(World_data["Years"], *param)

plt.figure(figsize=(10, 6))
sns.lineplot(data=World_data, x="Years", y="World", label="World")
sns.lineplot(data=World_data, x="Years", y="trial", label="Exponential Fit")
plt.xlabel("Years")
plt.ylabel("CO2 Emissions (kt)")
plt.title("CO2 Emissions Over Time with Exponential Fit")
plt.legend()
plt.show()


# In[26]:


def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically.
    """
    
    # initiate sigma the same shape as parameter
    var = np.zeros_like(x)   # initialise variance vector
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respect to the jth parameter
            deriv2 = deriv(x, func, parameter, j)
            # multiplied with the i-jth covariance
            # variance vector 
            var = var + deriv1 * deriv2 * covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta. Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.
    """

    # create vector with zeros and insert delta value for the relevant parameter
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """
    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar / matrix
    
    return corr


# In[27]:


years_future = np.arange(2021, 2031, 1)
predictions = exponential(years_future, *param)
confidence_range = error_prop(years_future, exponential, param, covar)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=World_data["Years"], y=World_data["World"], label="Data")
sns.lineplot(x=years_future, y=predictions, label="Best Fitting Function", color='black')
sns.lineplot(x=years_future, y=predictions - confidence_range, color='blue', alpha=0.2, label="Confidence Range")
sns.lineplot(x=years_future, y=predictions + confidence_range, color='blue', alpha=0.2)
plt.xlabel("Years")
plt.ylabel("CO2 Emissions (kt)")
plt.title("Exponential Growth Prediction from 2020-2030")
plt.legend()
plt.show()


# In[ ]:




