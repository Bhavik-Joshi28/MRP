#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Read the fraud train CSV file
fraud_train_df = pd.read_csv(r'C:\Users\Bansii\Downloads\ADV_Project\fraudtrain.csv')

# Display the head of the fraud train dataset
print("Head of Fraud Train Dataset:")
print(fraud_train_df.head())

# Read the fraud test CSV file
fraud_test_df = pd.read_csv(r'C:\Users\Bansii\Downloads\ADV_Project\fraudtest.csv')

# Display the head of the fraud test dataset
print("\nHead of Fraud Test Dataset:")
print(fraud_test_df.head())


# In[2]:


# Count the number of records in each dataset
train_records = len(fraud_train_df)
test_records = len(fraud_test_df)

print("Number of records in Fraud Train Dataset:", train_records)
print("Number of records in Fraud Test Dataset:", test_records)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv", usecols=['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud'])

# Step 2: Handling Missing Values
fraud_train_df.fillna({'category': 'Unknown'}, inplace=True)  # Fill missing category values

# Step 3: Data Type Conversion
# No conversion needed for datetime

# Step 4: Encoding Categorical Variables
fraud_train_df = pd.get_dummies(fraud_train_df, columns=['category'])

# Step 5: Normalization/Scaling
scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
fraud_train_df[numerical_cols] = scaler.fit_transform(fraud_train_df[numerical_cols])

# Visualization 1: Distribution of Transaction Amounts
plt.figure(figsize=(10, 6))
sns.histplot(fraud_train_df['amt'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()

# Visualization 2: Fraud vs. Non-fraud Transactions
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=fraud_train_df)
plt.title('Count of Fraud vs. Non-fraud Transactions')
plt.show()

# Visualization 3: Transaction Amounts by Categories
category_cols = [col for col in fraud_train_df.columns if 'category_' in col]
fraud_train_df['total_amt_by_category'] = fraud_train_df[category_cols].multiply(fraud_train_df['amt'], axis="index").sum(axis=1)
sns.barplot(x=fraud_train_df[category_cols].sum().index, y=fraud_train_df[category_cols].sum().values)
plt.xticks(rotation=45)
plt.title('Transaction Amounts by Category')
plt.show()


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv", usecols=['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud'])

# Filter only fraud transactions
fraud_train_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Step 2: Handling Missing Values
fraud_train_df.fillna({'category': 'Unknown'}, inplace=True)  # Fill missing category values

# Step 3: Data Type Conversion
# No conversion needed for datetime

# Step 4: Encoding Categorical Variables
fraud_train_df = pd.get_dummies(fraud_train_df, columns=['category'])

# Step 5: Normalization/Scaling
scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
fraud_train_df[numerical_cols] = scaler.fit_transform(fraud_train_df[numerical_cols])

# Visualization 1: Distribution of Transaction Amounts
plt.figure(figsize=(10, 6))
sns.histplot(fraud_train_df['amt'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts (Fraudulent)')
plt.show()

# Visualization 2: Transaction Amounts by Categories
category_cols = [col for col in fraud_train_df.columns if 'category_' in col]
fraud_train_df['total_amt_by_category'] = fraud_train_df[category_cols].multiply(fraud_train_df['amt'], axis="index").sum(axis=1)
sns.barplot(x=fraud_train_df[category_cols].sum().index, y=fraud_train_df[category_cols].sum().values)
plt.xticks(rotation=45)
plt.title('Transaction Amounts by Category (Fraudulent)')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Step 2: Exploratory Data Analysis (EDA)
# Summary Statistics
print("Summary Statistics:")
print(fraud_train_df.describe())

# Data Distributions
# Distribution of Transaction Amounts
plt.figure(figsize=(10, 6))
sns.histplot(fraud_train_df['amt'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Filter only fraud transactions
fraud_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Step 2: Exploratory Data Analysis (EDA) for Fraud Transactions
# Summary Statistics
print("Summary Statistics for Fraud Transactions:")
print(fraud_transactions_df.describe())

# Data Distributions
# Distribution of Transaction Amounts for Fraud Transactions
plt.figure(figsize=(10, 6))
sns.histplot(fraud_transactions_df['amt'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts for Fraud Transactions')
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Step 2: Data Preparation
# Extract month from the transaction date
fraud_train_df['trans_month'] = pd.to_datetime(fraud_train_df['trans_date_trans_time']).dt.month

# Filter only fraud transactions
fraud_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]


# # What month of the year has the highest number of transactions take place?  
# 

# In[7]:


import calendar

# Step 3: Analysis
# Count the number of fraudulent transactions per month
fraud_transactions_by_month = fraud_transactions_df['trans_month'].value_counts().sort_index()

# Step 4: Visualization
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=fraud_transactions_by_month.index, y=fraud_transactions_by_month.values, palette='viridis')
plt.title('Number of Fraudulent Transactions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')

# Add value annotations to the top of each bar
for i, value in enumerate(fraud_transactions_by_month.values):
    bar_plot.text(i, value + 5, str(value), ha='center', color='black')

plt.xticks(range(12), [calendar.month_abbr[i+1] for i in range(12)])  # Corrected month labels
plt.show()




# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")

# Step 2: Data Preparation
# Extract month from the transaction date
fraud_train_df['trans_month'] = pd.to_datetime(fraud_train_df['trans_date_trans_time']).dt.month

# Filter only fraud transactions
fraud_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Create a pivot table to aggregate the count of fraudulent transactions by month
pivot_table = fraud_transactions_df.pivot_table(index='trans_month', aggfunc='size')

# Reshape the pivot table for heatmap
heatmap_data = pivot_table.reset_index(name='fraud_count').pivot(index='trans_month', columns='fraud_count', values='fraud_count')

# Replace month numbers with month names
heatmap_data.index = heatmap_data.index.map(lambda x: calendar.month_abbr[x])

# Plotting the heatmap with reversed colormap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='viridis_r', annot=True, fmt='g', linewidths=0.5)
plt.title('Number of Fraudulent Transactions by Month')
plt.xlabel('Fraud Count')
plt.ylabel('Month')
plt.show()



# 
# Based on the visualization of fraudulent transactions by month:
# 
# The month with the highest number of fraudulent transactions is January.
# Following January, February and March also show relatively high numbers of fraudulent transactions.
# The months with the lowest number of fraudulent transactions are May, June, and July.

# # What are the top five states that have the highest amount of fraudulent transactions overall in terms of TOTAL $ AMOUNT IN FRAUD? 

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Step 2: Data Preparation
# Filter only fraudulent transactions
fraudulent_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Calculate total fraudulent transaction amount per state
total_fraud_per_state = fraudulent_transactions_df.groupby('state')['amt'].sum().sort_values(ascending=False).head(5)

# Step 3: Visualization
plt.figure(figsize=(10, 6))

# Use a seaborn color palette for better color selection
colors = sns.color_palette("Blues_r", len(total_fraud_per_state))

# Horizontal bar plot
bars = plt.barh(total_fraud_per_state.index, total_fraud_per_state.values, color=colors)

# Show amounts on the right of each bar
for bar, value in zip(bars, total_fraud_per_state.values):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value/1000:.1f}K', va='center', fontsize=10)

plt.title('Top 5 States with Highest Fraudulent Transaction Amount')
plt.xlabel('Total Fraud Amount ($)')
plt.ylabel('State')
plt.tight_layout()
plt.show()




# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import squarify  # Library for treemap visualization

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")

# Step 2: Data Preparation
# Filter only fraudulent transactions
fraudulent_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Calculate total fraudulent transaction amount per state
total_fraud_per_state = fraudulent_transactions_df.groupby('state')['amt'].sum().sort_values(ascending=False).head(5)

# Print state names and amounts
for state, amount in total_fraud_per_state.items():
    print(f"{state}: ${amount:,.2f}")

# Step 3: Visualization (Treemap)
plt.figure(figsize=(12, 8))

# Define a reversed colormap with shades of blue
color_map = plt.cm.Blues_r

# Generate colors based on the total fraud amount
colors = [color_map(i / len(total_fraud_per_state)) for i in range(len(total_fraud_per_state))]

# Plot the treemap
squarify.plot(sizes=total_fraud_per_state.values, label=total_fraud_per_state.index, color=colors, alpha=0.7)

plt.title('Top 5 States with Highest Fraudulent Transaction Amount (Treemap)')
plt.axis('off')  # Turn off axis

plt.show()



# # What are the top 3 professions most associated with fraudulent transactions overall?

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Step 2: Data Preparation
# Filter only fraudulent transactions
fraudulent_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Calculate the frequency of fraudulent transactions per profession
fraudulent_transactions_per_profession = fraudulent_transactions_df['job'].value_counts().sort_values(ascending=False).head(3)

# Step 3: Visualization (Bar Chart)
plt.figure(figsize=(10, 6))

# Choose lighter color palette for the plot
colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Light green, Light orange, Light blue

# Plot bar chart
bars = plt.bar(fraudulent_transactions_per_profession.index, fraudulent_transactions_per_profession.values, color=colors)

# Show values on top of each bar
for bar, value in zip(bars, fraudulent_transactions_per_profession.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value}', ha='center', va='bottom', fontsize=10)

plt.title('Top 3 Professions Associated with Fraudulent Transactions')
plt.xlabel('Profession')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()



# In[12]:


import pandas as pd
import plotly.express as px

# Step 1: Data Loading
fraud_train_df = pd.read_csv("fraudtrain.csv")
fraud_test_df = pd.read_csv("fraudtest.csv")

# Step 2: Data Preparation
# Filter only fraudulent transactions
fraudulent_transactions_df = fraud_train_df[fraud_train_df['is_fraud'] == 1]

# Step 3: Count occurrences of each profession
top_professions = fraudulent_transactions_df['job'].value_counts().head(3)

# Step 4: Visualization with Plotly
fig = px.pie(names=top_professions.index, values=top_professions.values, title='Top 3 Professions Most Associated with Fraudulent Transactions',
             color_discrete_sequence=px.colors.qualitative.Pastel1)  # Change colors for better visibility
fig.update_traces(hoverinfo='label+percent+value', hovertemplate='%{label}: %{value}', textinfo='label+percent+value', textfont_color='black')  # Customize hover information and text color
fig.show()



# Based on the provided output, the top 3 professions associated with fraudulent transactions, along with their respective counts, are:
# 
# Materials engineer: 62
# Trading standards officer: 56
# Naval architect: 53
# 
# 

# # What are the top 5 merchants that have the highest number of fraudulent transactions overall? 

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Provided data
merchant_data = {
    'Merchant': ['fraud_Rau and Sons', 'fraud_Kozey-Boehm', 'fraud_Cormier LLC', 'fraud_Doyle Ltd', 'fraud_Vandervort-Funk'],
    'Transactions': [49, 48, 48, 47, 47]
}

# Convert data to DataFrame
merchant_df = pd.DataFrame(merchant_data)

# Sort DataFrame by Transactions in descending order
merchant_df = merchant_df.sort_values(by='Transactions', ascending=True)

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Transactions', y='Merchant', data=merchant_df, palette='Blues_r')

# Add data labels
for index, value in enumerate(merchant_df['Transactions']):
    plt.text(value, index, str(value), va='center')

# Add labels and title
plt.xlabel('Number of Transactions')
plt.ylabel('Merchant')
plt.title('Top 5 Merchants with the Highest Number of Fraudulent Transactions')
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.show()




# In[14]:


import pandas as pd
import folium
import random

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Calculate the number of fraudulent transactions for each merchant
fraud_counts = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1].groupby('merchant').size()

# Sort the counts in descending order to get the top 5 merchants with the most fraudulent transactions
top_merchants = fraud_counts.sort_values(ascending=False).head(5)

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Define colors for each merchant
colors = ['#FF5733', '#33FF57', '#5733FF', '#FF33A8', '#33A8FF']

# Print out the top 5 merchants with a dot of each color
for index, (merchant, _) in enumerate(top_merchants.items()):
    color = colors[index % len(colors)]  # Choose color from the defined list
    print(f"\x1b[38;2;{int(color[1:3], 16)};{int(color[3:5], 16)};{int(color[5:], 16)}m\u25CF\x1b[0m Merchant: {merchant}")

# Add markers for each of the top 5 merchant locations
for index, (merchant, fraud) in enumerate(top_merchants.items()):
    merchant_locations = fraud_transactions_df[(fraud_transactions_df['merchant'] == merchant) & (fraud_transactions_df['is_fraud'] == 1)]
    color = colors[index % len(colors)]  # Choose color from the defined list
    for _, row in merchant_locations.iterrows():
        popup_text = f"Merchant: {merchant}<br>Fraudulent Transactions: {fraud}"
        folium.CircleMarker(location=[row['merch_lat'], row['merch_long']], radius=3, color=color, fill=True, fill_color=color, popup=popup_text).add_to(m)

# Display the map
m


# 
# Top 5 Merchants with the Highest Number of Fraudulent Transactions:
# merchant
# fraud_Rau and Sons    :      49
# fraud_Kozey-Boehm     :      48
# fraud_Cormier LLC     :      48
# fraud_Doyle Ltd       :      47
# fraud_Vandervort-Funk :      4

# # Comparing client location and merchant’s location using longitude and latitude to figure out which area in a given state has the highest fraud density as per given location?

# In[15]:


import pandas as pd
import folium

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[(fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)]

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Add markers for each fraudulent transaction, with different colors based on the state
for index, row in gas_transport_fraud_transactions.iterrows():
    folium.CircleMarker(location=[row['lat'], row['long']], radius=3, color='red', fill=True, fill_color='red').add_to(m)

# Display the map
m


# In[16]:


import pandas as pd
import folium

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[(fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)]

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Add markers for each fraudulent transaction, with different colors based on gender and state
for index, row in gas_transport_fraud_transactions.iterrows():
    color = 'red' if row['gender'] == 'F' else 'blue'
    folium.CircleMarker(location=[row['lat'], row['long']], radius=3, color=color, fill=True, fill_color=color).add_to(m)

# Display the map
m



# In[17]:


import pandas as pd
import folium

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[(fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)]

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Add markers for client locations (in blue)
for _, row in gas_transport_fraud_transactions.iterrows():
    folium.CircleMarker(location=[row['lat'], row['long']], radius=3, color='blue', fill=True, fill_color='blue').add_to(m)

# Add markers for merchant locations (in yellow)
for _, row in gas_transport_fraud_transactions.iterrows():
    folium.CircleMarker(location=[row['merch_lat'], row['merch_long']], radius=3, color='yellow', fill=True, fill_color='yellow').add_to(m)

# Display the map
m


# In[18]:


from scipy.spatial import cKDTree
import numpy as np

# Extract coordinates of yellow (merchant) and blue (client) points
yellow_points = gas_transport_fraud_transactions[['merch_lat', 'merch_long']].values
blue_points = gas_transport_fraud_transactions[['lat', 'long']].values

# Build KD-Tree for blue (client) points
blue_tree = cKDTree(blue_points)

# Query KD-Tree to find nearest blue (client) point for each yellow (merchant) point
distances, _ = blue_tree.query(yellow_points)

# Calculate the average distance between nearest yellow and blue points
average_distance_yellow_blue = np.mean(distances)

print("Average distance between nearest yellow (merchant) and blue (client) points (km):", average_distance_yellow_blue)



# 
# Findings:
# 
# Distribution of Fraudulent Transactions by Month:
# 
# January has the highest number of fraudulent transactions, followed by February and March.
# May, June, and July exhibit the lowest number of fraudulent transactions.
# 
# Top States with Highest Fraudulent Transaction Amounts:
# 
# New York (NY), Texas (TX), Pennsylvania (PA), California (CA), and Ohio (OH) have the highest total dollar amount in fraudulent transactions.
# 
# Professions Most Associated with Fraudulent Transactions:
# 
# Materials engineers, trading standards officers, and naval architects are the top three professions associated with fraudulent transactions based on counts.
# 
# Top Merchants with Highest Number of Fraudulent Transactions:
# 
# The top five merchants with the highest number of fraudulent transactions are "fraud_Rau and Sons," "fraud_Kozey-Boehm," "fraud_Cormier LLC," "fraud_Doyle Ltd," and "fraud_Vandervort-Funk."
# 
# Spatial Patterns of Fraudulent Transactions:
# 
# The average distance between nearest merchant (yellow) and client (blue) points is approximately 0.5 kilometers (km).
# 
# 
# Results:
# 
# January sees the highest frequency of fraudulent transactions, suggesting a potential seasonal trend or increased fraudulent activity at the beginning of the year.
# New York, Texas, and Pennsylvania emerge as hotspots for fraudulent transactions based on the total dollar amount, indicating the need for targeted fraud prevention measures in these states.
# Certain professions, such as materials engineering, trading standards, and naval architecture, exhibit higher associations with fraudulent activities, highlighting potential areas for further investigation and monitoring.
# Specific merchants, identified through their merchant names, are responsible for a significant number of fraudulent transactions, necessitating enhanced scrutiny and oversight in their operations.
# The close proximity between merchants and clients involved in fraudulent transactions, as indicated by the average distance of 0.5 km, suggests potential collusion or local fraud networks operating in concentrated geographic areas.
# 
# 
# Conclusions:
# 
# Financial institutions should allocate resources towards implementing robust fraud detection and prevention mechanisms, particularly during peak periods of fraudulent activity such as January.
# Enhanced monitoring and surveillance efforts are warranted in states like New York, Texas, and Pennsylvania, where fraudulent transaction amounts are highest.
# Authorities should investigate the involvement of certain professions, such as materials engineers and trading standards officers, in fraudulent activities and implement measures to deter such behavior.
# Close scrutiny and regulatory oversight should be directed towards merchants identified as top contributors to fraudulent transactions to mitigate their impact on financial systems.
# Geospatial analysis can provide valuable insights into the spatial distribution of fraudulent activities, aiding in the identification of localized fraud networks and informing targeted intervention strategies.
# 
# 

# # Predictive Analytics

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the original data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category
gas_transport_transactions = fraud_transactions_df[fraud_transactions_df['category'] == 'gas_transport']

# Calculate fraud density based on client and merchant locations
# Here, you should calculate the density based on the number of transactions within a certain radius around each location
# For simplicity, let's assume a fixed radius and count the number of transactions within that radius for each location

radius = 0.5  # Assume a radius of 0.5 km

# Group by location and count the number of transactions within the radius
fraud_density_data = gas_transport_transactions.groupby(['lat', 'long', 'merch_lat', 'merch_long']).size().reset_index(name='total_transactions')

# Merge the transaction count with the original data
merged_data = pd.merge(gas_transport_transactions, fraud_density_data, on=['lat', 'long', 'merch_lat', 'merch_long'], how='left')

# Define features and target variable
X = merged_data[['lat', 'long', 'merch_lat', 'merch_long', 'total_transactions']]
y = merged_data['is_fraud']

# Impute missing values in features with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Training Accuracy:", train_score)
print("Testing Accuracy:", test_score)


# In[20]:


import pandas as pd
import folium

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[(fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)]

# Group by city and calculate the average population for each city
city_population_avg = fraud_transactions_df.groupby('city')['city_pop'].mean()

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Add markers for client locations (in blue)
for _, row in gas_transport_fraud_transactions.iterrows():
    folium.CircleMarker(location=[row['lat'], row['long']], radius=3, color='blue', fill=True, fill_color='blue').add_to(m)

# Add markers for merchant locations (in yellow)
for _, row in gas_transport_fraud_transactions.iterrows():
    folium.CircleMarker(location=[row['merch_lat'], row['merch_long']], radius=3, color='yellow', fill=True, fill_color='yellow').add_to(m)

# Add markers for cities with population greater than 62400 (in black)
for city, population in city_population_avg.items():
    if population > 62400:
        city_coords = fraud_transactions_df.loc[fraud_transactions_df['city'] == city, ['lat', 'long']].iloc[0]
        folium.CircleMarker(location=[city_coords['lat'], city_coords['long']], radius=5, color='black', fill=True, fill_color='red').add_to(m)

# Display the map
m


# In[21]:


import pandas as pd
import folium
from sklearn.cluster import KMeans

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[(fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)]

# Group by city and calculate the average population for each city
city_population_avg = fraud_transactions_df.groupby('city')['city_pop'].mean()

# Perform clustering on gas station charges locations
gas_station_locations = gas_transport_fraud_transactions[['lat', 'long']].values
kmeans_gas = KMeans(n_clusters=5, random_state=42).fit(gas_station_locations)
gas_clusters_centers = kmeans_gas.cluster_centers_

# Perform clustering on client locations
client_locations = fraud_transactions_df[['lat', 'long']].values
kmeans_clients = KMeans(n_clusters=5, random_state=42).fit(client_locations)
client_clusters_centers = kmeans_clients.cluster_centers_

# Create a map centered on the US
m = folium.Map(location=[37, -95], zoom_start=4)

# Add markers for gas station charge clusters (in blue)
for center in gas_clusters_centers:
    folium.CircleMarker(location=center, radius=10, color='blue', fill=True, fill_color='blue').add_to(m)

# Add markers for client location clusters (in red)
for center in client_clusters_centers:
    folium.CircleMarker(location=center, radius=10, color='red', fill=True, fill_color='red').add_to(m)

# Add markers for cities with population greater than 62400 (in black)
for city, population in city_population_avg.items():
    if population > 62400:
        city_coords = fraud_transactions_df.loc[fraud_transactions_df['city'] == city, ['lat', 'long']].iloc[0]
        folium.CircleMarker(location=[city_coords['lat'], city_coords['long']], radius=5, color='black', fill=True, fill_color='black').add_to(m)

# Display the map
m


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category
gas_transport_transactions = fraud_transactions_df[fraud_transactions_df['category'] == 'gas_transport']

# Perform clustering on gas station charges locations
gas_station_locations = gas_transport_transactions[['lat', 'long']].values
kmeans_gas = KMeans(n_clusters=5, random_state=42).fit(gas_station_locations)
gas_clusters_centers = kmeans_gas.cluster_centers_

# Calculate distance to high-risk areas
gas_transport_transactions['distance_to_high_risk'] = gas_transport_transactions.apply(lambda row: min([geodesic((row['lat'], row['long']), center).km for center in gas_clusters_centers]), axis=1)

# Calculate transaction density
transaction_density = gas_transport_transactions.groupby(['lat', 'long']).size().reset_index(name='transaction_density')
gas_transport_transactions = pd.merge(gas_transport_transactions, transaction_density, on=['lat', 'long'], how='left')

# Define features and target variable
X = gas_transport_transactions[['lat', 'long', 'merch_lat', 'merch_long', 'distance_to_high_risk', 'transaction_density']]
y = gas_transport_transactions['is_fraud']

# Impute missing values in features with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Training Accuracy:", train_score)
print("Testing Accuracy:", test_score)




# With a training accuracy of 99.999% and a testing accuracy of 99.575%, it suggests that the model is performing very well in distinguishing between fraud and non-fraudulent charges within the high-risk cluster. This indicates that the features used to define the high-risk cluster are effective in capturing the characteristics of fraudulent transactions within the gas transport category. Therefore, it's reasonable to conclude that the identified high-risk cluster is indeed associated with a higher likelihood of fraudulent activit

# 
# Top 3 merchant for gas station charges where fraud reported.

# In[23]:


import pandas as pd

# Load the data (assuming the dataset is already loaded as fraud_transactions_df)
# If not loaded, uncomment the next line:
# fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[
    (fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)
]

# Count fraudulent transactions per merchant
fraud_per_merchant = gas_transport_fraud_transactions['merchant'].value_counts()

# Display top 3 merchants with the highest number of fraud reports
top_3_fraud_merchants = fraud_per_merchant.head(3)
print("Top 3 gas station merchants with the highest number of fraud reports:")
print(top_3_fraud_merchants)


# In[24]:


import pandas as pd
import folium

# Load the data from the file path
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[
    (fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)
]

# Count fraudulent transactions per merchant and get the top 3
fraud_per_merchant = gas_transport_fraud_transactions['merchant'].value_counts()
top_3_fraud_merchants = fraud_per_merchant.head(3)

# Create a map centered around an average location (this might need adjustment based on your data's geography)
m = folium.Map(location=[37, -95], zoom_start=4)

# Define a fixed radius for all markers
marker_radius = 5

# Assign colors to each of the top merchants
merchant_colors = {
    top_3_fraud_merchants.index[0]: 'red',
    top_3_fraud_merchants.index[1]: 'blue',
    top_3_fraud_merchants.index[2]: 'yellow'
}

# Add markers for top merchant locations
for merchant, color in merchant_colors.items():
    merchant_transactions = gas_transport_fraud_transactions[gas_transport_fraud_transactions['merchant'] == merchant]
    for _, row in merchant_transactions.iterrows():
        folium.CircleMarker(
            location=[row['merch_lat'], row['merch_long']],
            radius=marker_radius,
            color=color,
            fill=True,
            fill_color=color,
            popup=merchant
        ).add_to(m)

# Filter client locations who reported fraud for the top 3 gas station merchants
client_locations = gas_transport_fraud_transactions[['lat', 'long']].values

# Add markers for client locations (in black)
for location in client_locations:
    folium.CircleMarker(
        location=location,
        radius=marker_radius,
        color='black',
        fill=True,
        fill_color='black',
        popup="Client"
    ).add_to(m)

# Display the map
m.save("fraud_map.html")
m
import pandas as pd
import folium

# Load the data from the file path
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category and fraud charges
gas_transport_fraud_transactions = fraud_transactions_df[
    (fraud_transactions_df['category'] == 'gas_transport') & (fraud_transactions_df['is_fraud'] == 1)
]

# Count fraudulent transactions per merchant and get the top 3
fraud_per_merchant = gas_transport_fraud_transactions['merchant'].value_counts()
top_3_fraud_merchants = fraud_per_merchant.head(3).index.tolist()

# Assign colors to each of the top merchants
merchant_colors = {
    top_3_fraud_merchants[0]: 'red',
    top_3_fraud_merchants[1]: 'blue',
    top_3_fraud_merchants[2]: 'yellow'
}

# Create a map centered around an average location (this might need adjustment based on your data's geography)
m = folium.Map(location=[37, -95], zoom_start=4)

# Define a fixed radius for all markers
marker_radius = 5

# Add markers for top merchant locations
for _, row in gas_transport_fraud_transactions.iterrows():
    if row['merchant'] in top_3_fraud_merchants:
        folium.CircleMarker(
            location=[row['merch_lat'], row['merch_long']],
            radius=marker_radius,
            color=merchant_colors[row['merchant']],
            fill=True,
            fill_color=merchant_colors[row['merchant']],
            popup=row['merchant']
        ).add_to(m)

# Add markers for client locations (in black) who reported the fraud charges
for _, row in gas_transport_fraud_transactions.iterrows():
    if row['merchant'] in top_3_fraud_merchants:
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=marker_radius,
            color='black',
            fill=True,
            fill_color='black',
            popup="Client: " + str(row['cc_num'])  # Assuming cc_num can uniquely identify a client
        ).add_to(m)

# Display the map
m.save("fraud_map.html")
m


# Top 3 gas station merchants with the highest number of fraud reports:
# merchant, 
# fraud_Prohaska-Murray                  Red , 
# fraud_Koss and Sons                    Blue ,
# fraud_Eichmann, Bogan and Rodriguez    Yellow , 
# Black color shows client's location who reported fraud for listed charges.

# By looking at this we can directly conclude that Gas station merchant Prohaska-Murray, Koss and Sons,
# Eichmann, Bogan and Rodriguez has highest risk of fraud at the costal side of US on east and west both. And merchant Prohaska-Murray, Koss and Sons, has fraudulent activity at the gas station on all over the US but merchant Eichmann, Bogan and Rodriguez has higher fraudlent activity on the righthand side the USA states.  

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for the "gas_transport" category
gas_transport_transactions = fraud_transactions_df[fraud_transactions_df['category'] == 'gas_transport']

# Perform clustering on gas station charges locations
gas_station_locations = gas_transport_transactions[['lat', 'long']].values
kmeans_gas = KMeans(n_clusters=5, random_state=42).fit(gas_station_locations)
gas_clusters_centers = kmeans_gas.cluster_centers_

# Calculate distance to high-risk areas
gas_transport_transactions['distance_to_high_risk'] = gas_transport_transactions.apply(
    lambda row: min([geodesic((row['lat'], row['long']), center).km for center in gas_clusters_centers]), axis=1
)

# Calculate transaction density
transaction_density = gas_transport_transactions.groupby(['lat', 'long']).size().reset_index(name='transaction_density')
gas_transport_transactions = pd.merge(gas_transport_transactions, transaction_density, on=['lat', 'long'], how='left')

# Define features and target variable
X = gas_transport_transactions[['lat', 'long', 'merch_lat', 'merch_long', 'distance_to_high_risk', 'transaction_density']]
y = gas_transport_transactions['is_fraud']

# Impute missing values in features with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Training Accuracy:", train_score)
print("Testing Accuracy:", test_score)

# Generate confusion matrix and classification report
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot Learning Curves
train_sizes, train_scores, test_scores = learning_curve(model, X_imputed, y, cv=5, n_jobs=-1)
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='g', label='Cross-validation score')
plt.title('Learning Curves (Random Forest)')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()

# Plot Validation Curves for n_estimators
param_range = range(1, 201, 10)
train_scores, test_scores = validation_curve(
    RandomForestClassifier(random_state=42), X_imputed, y, param_name='n_estimators',
    param_range=param_range, scoring='accuracy', n_jobs=-1
)
plt.figure(figsize=(12, 6))
plt.plot(param_range, train_scores.mean(axis=1), 'o-', color='r', label='Training score')
plt.plot(param_range, test_scores.mean(axis=1), 'o-', color='g', label='Cross-validation score')
plt.title('Validation Curve with Random Forest')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()


# model is likely overfitting. 
# 
# Let's break down the results to understand why:
# 
# Evaluation Metrics Breakdown
# Training Accuracy: 99.999%
# 
# This is extremely high, indicating that the model is performing exceptionally well on the training data.
# Testing Accuracy: 99.575%
# 
# This is also very high, but slightly lower than the training accuracy. This discrepancy, while small, can still suggest overfitting.
# Confusion Matrix:
# 
# True Negatives (TN): 26,214
# False Positives (FP): 0
# False Negatives (FN): 112
# True Positives (TP): 6
# Classification Report:
# 
# Precision for class 0 (non-fraud): 1.00
# Recall for class 0 (non-fraud): 1.00
# F1-score for class 0 (non-fraud): 1.00
# Precision for class 1 (fraud): 1.00
# Recall for class 1 (fraud): 0.05
# F1-score for class 1 (fraud): 0.10
# Interpretation
# Overfitting Indicators:
# 
# High Training Accuracy: The model performs almost perfectly on the training set. This is often a sign that the model has learned the training data too well, potentially capturing noise and outliers.
# Low Recall for Fraud Class: The recall for the fraud class (1) is very low (0.05). This means the model is not identifying most of the actual fraud cases correctly. Even though precision is high for fraud (1), it is because the model is making very few predictions for this class (low recall). This imbalance is a strong indicator of overfitting, as the model is biased towards the majority class (non-fraud).
# Why Overfitting Happens:
# 
# Class Imbalance: Your dataset likely has a large imbalance between the number of fraud and non-fraud cases. The model may have learned to predict the majority class (non-fraud) very well, but struggles with the minority class (fraud). This imbalance can cause overfitting, as the model becomes overly specialized in predicting the majority class.
# Model Complexity: Random Forest models with many trees and deep trees can capture complex patterns in the data, but if the data is not sufficiently diverse or is imbalanced, the model can overfit to the training data, leading to poor generalization.

# In[ ]:





# In[26]:


import pandas as pd

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for fraud charges
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Number of fraud transactions
num_fraud_transactions = fraud_transactions.shape[0]

# Number of unique clients who reported fraud
num_unique_clients = fraud_transactions['cc_num'].nunique()

# Number of unique merchants involved in fraud
num_unique_merchants = fraud_transactions['merchant'].nunique()

# Print the results
print(f"Number of Fraud Transactions: {num_fraud_transactions}")
print(f"Number of Unique Clients Who Reported Fraud: {num_unique_clients}")
print(f"Number of Unique Merchants Involved in Fraud: {num_unique_merchants}")


# In[27]:


import pandas as pd

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Total number of transactions in the database
total_transactions = fraud_transactions_df.shape[0]

# Number of fraud transactions
num_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1].shape[0]

# Number of non-fraud transactions
num_non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].shape[0]

# Print the results
print(f"Total Number of Transactions: {total_transactions}")
print(f"Number of Fraud Transactions: {num_fraud_transactions}")
print(f"Number of Non-Fraud Transactions: {num_non_fraud_transactions}")


# In[28]:


import sklearn
import imblearn

print(f"scikit-learn version: {sklearn.__version__}")
print(f"imbalanced-learn version: {imblearn.__version__}")


# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert y to a numpy array
y = np.array(y)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"Class distribution in training set: {class_distribution}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Create and train Random Forest model with class weights
model_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = model_rf.predict(X_test)
print("Random Forest with Class Weights:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(model_rf, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest with Class Weights:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')



# Balanced Dataset Creation:
# 
# Action: Filtered the data to include an equal number of fraudulent and non-fraudulent transactions.
# Purpose: Reduces bias in the model by ensuring that both classes are equally represented in the dataset.
# Feature Preparation and Imputation:
# 
# Action: Selected features and used mean imputation to handle missing values.
# Purpose: Ensures that the model can handle incomplete data without introducing bias.
# Stratified Data Splitting:
# 
# Action: Split the data into training and testing sets using stratification.
# Purpose: Maintains the class distribution in both the training and testing sets, providing a better estimate of model performance and reducing bias.
# Class Weight Adjustment:
# 
# Action: Computed and applied class weights to address class imbalance in the training process.
# Purpose: Gives more importance to the minority class (fraudulent transactions) during training, improving the model's ability to detect less frequent events.
# Model Evaluation:
# 
# Action: Evaluated the model using classification metrics (precision, recall, F1-score) and confusion matrix.
# Purpose: Assesses how well the model generalizes to unseen data, identifying any issues with overfitting or underfitting.
# Cross-Validation:
# 
# Action: Performed cross-validation to evaluate the model’s performance across different data subsets.
# Purpose: Provides a more reliable estimate of model performance and helps in detecting overfitting by assessing the model’s consistency across various data partitions.
# 
# These steps are designed to ensure that the Random Forest model performs well on unseen data by addressing issues such as class imbalance, missing values, and overfitting. The combination of balancing the dataset, handling missing values, and using cross-validation helps in building a robust and generalizable model.

# In[30]:


import pandas as pd
import folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert y to a numpy array
y = np.array(y)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"Class distribution in training set: {class_distribution}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Create and train Random Forest model with class weights
model_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = model_rf.predict(X_test)
print("Random Forest with Class Weights:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
scores_rf = cross_val_score(model_rf, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest with Class Weights:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')

# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Save the map as an HTML file
map_folium.save("fraud_transactions_map.html")

print("Map has been saved as 'fraud_transactions_map.html'")


# In[ ]:





# In[31]:


import pandas as pd
import numpy as np
import folium
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert y to a numpy array
y = np.array(y)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"Class distribution in training set: {class_distribution}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Create and train Random Forest model with class weights
model_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = model_rf.predict(X_test)
print("Random Forest with Class Weights:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
scores_rf = cross_val_score(model_rf, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest with Class Weights:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')

# Plot Learning Curve
train_sizes, train_scores, validation_scores = learning_curve(
    model_rf, 
    X_imputed, 
    y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Compute mean and standard deviation for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, validation_mean, 'o-', color='g', label='Validation score')

# Plot the fill between for standard deviation
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='g', alpha=0.1)

plt.xlabel('Training Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid()
plt.show()

# Plot Validation Curve
param_range = [10, 50, 100, 150, 200]
train_scores, validation_scores = validation_curve(
    model_rf, 
    X_imputed, 
    y, 
    param_name='n_estimators',
    param_range=param_range,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Compute mean and standard deviation for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(12, 6))
plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
plt.plot(param_range, validation_mean, 'o-', color='g', label='Validation score')

# Plot the fill between for standard deviation
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(param_range, validation_mean - validation_std, validation_mean + validation_std, color='g', alpha=0.1)

plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score')
plt.title('Validation Curve')
plt.legend(loc='best')
plt.grid()
plt.show()

# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Save the map as an HTML file
map_folium.save("fraud_transactions_map.html")

print("Map has been saved as 'fraud_transactions_map.html'")



# In[ ]:





# In[32]:


import pandas as pd
import folium
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from IPython.display import display

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert y to a numpy array
y = np.array(y)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"Class distribution in training set: {class_distribution}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Create and train Random Forest model with class weights
model_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
model_rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_rf = model_rf.predict(X_test)
print("Random Forest with Class Weights:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
scores_rf = cross_val_score(model_rf, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest with Class Weights:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')

# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Display the map in the Jupyter Notebook
display(map_folium)


# 1. Class Distribution in Training Set:
# {0: 6004, 1: 6005} indicates that the training set is balanced with nearly equal numbers of non-fraudulent (class 0) and fraudulent (class 1) transactions. There are 6004 non-fraudulent and 6005 fraudulent transactions. This balance is achieved by undersampling the non-fraudulent class to match the number of fraudulent transactions, ensuring the classifier is not biased toward the majority class.
# 2. Classification Report:
# Precision:
# Precision for class 0 (non-fraudulent): 0.63 indicates that 63% of the transactions predicted as non-fraudulent are actually non-fraudulent.
# Precision for class 1 (fraudulent): 0.62 indicates that 62% of the transactions predicted as fraudulent are actually fraudulent.
# Recall:
# Recall for class 0: 0.60 indicates that 60% of the actual non-fraudulent transactions were correctly identified by the model.
# Recall for class 1: 0.64 indicates that 64% of the actual fraudulent transactions were correctly identified by the model.
# F1-Score:
# F1-score is the harmonic mean of precision and recall. For class 0, it is 0.61, and for class 1, it is 0.63, indicating a moderate balance between precision and recall for both classes.
# Support:
# Support refers to the number of true instances for each class. Here, both class 0 and class 1 have nearly equal support of 1502 and 1501, respectively, indicating a balanced test set as well.
# 3. Confusion Matrix:
# The confusion matrix provides a summary of the model's predictions:
# True Positives (TP): 961 transactions were correctly identified as fraudulent (class 1).
# True Negatives (TN): 906 transactions were correctly identified as non-fraudulent (class 0).
# False Positives (FP): 596 transactions were incorrectly identified as fraudulent (but were actually non-fraudulent).
# False Negatives (FN): 540 transactions were incorrectly identified as non-fraudulent (but were actually fraudulent).
# 4. Cross-Validation F1 Scores:
# The F1 scores obtained from 5-fold cross-validation are relatively low, ranging from 0.10 to 0.19, with an average F1 score of 0.1437.
# Interpretation: The low F1 scores suggest that the model might not generalize well across different data splits. It indicates that the classifier might be struggling to distinguish between the classes effectively, possibly due to the inherent complexity of the data or limitations in feature representation.
# 5. Model Visualization:
# The model visualization (folium map) plots the fraudulent and non-fraudulent transactions geographically.
# Red Markers: Represent fraudulent transactions.
# Blue Markers: Represent non-fraudulent transactions.
# Usage: The map visualization helps to understand the geographical distribution of fraud and non-fraud transactions, which could provide insights into potential fraud hotspots.
# Visualization Code Explanation:
# The map is generated using the folium library, where:
# 
# map_folium = folium.Map(location=map_center, zoom_start=12): Initializes the map centered around the average latitude and longitude of the transactions.
# Markers: Fraudulent and non-fraudulent transactions are plotted as red and blue markers, respectively, with popup information showing their coordinates.
# Conclusion:
# The model shows moderate performance with an overall accuracy of 62% and relatively balanced precision and recall. However, the low cross-validation F1 scores indicate that the model might not be robust across different data splits, suggesting a need for further feature engineering, model tuning, or exploration of more complex models to improve performance.
# This code and explanation provide a comprehensive overview of the Random Forest model's performance and its application to fraud detection with visual geographic insights.

# # Changing Approche to get more accurate results. 

# 
# 
# Start with Feature Engineering:
# 
# Implement time-based and distance features first, as these are likely to capture critical patterns in the data that can improve model performance.
# Follow up with Model Tuning:
# 
# Perform hyperparameter tuning on your Random Forest model using Grid Search or Random Search with cross-validation.
# Handle Imbalanced Data:
# 
# Apply SMOTE to further address any class imbalance issues that may persist.
# Optimize the Decision Threshold:
# 
# After retraining the model with the new features and parameters, adjust the decision threshold to find the optimal balance between precision and recall.
# Evaluate Results:
# 
# 
# This approach should provide a structured path to improving the model's performance, addressing both feature inadequacies and potential model tuning issues.
# 

# In[42]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Feature Engineering
fraud_transactions_df['trans_date_trans_time'] = pd.to_datetime(fraud_transactions_df['trans_date_trans_time'])
fraud_transactions_df['transaction_hour'] = fraud_transactions_df['trans_date_trans_time'].dt.hour
fraud_transactions_df['transaction_day'] = fraud_transactions_df['trans_date_trans_time'].dt.dayofweek
fraud_transactions_df['transaction_month'] = fraud_transactions_df['trans_date_trans_time'].dt.month

# Calculate distance from home or usual location
def calculate_distance(row):
    return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km

fraud_transactions_df['distance_from_home'] = fraud_transactions_df.apply(calculate_distance, axis=1)

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long', 'transaction_hour', 'transaction_day', 'transaction_month', 'distance_from_home']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Define the Random Forest model and hyperparameters for Grid Search
model_rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

best_rf_model = grid_search.best_estimator_

# Predict and evaluate the model
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

print("Best Random Forest Model:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(best_rf_model, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Save the map as an HTML file
map_folium.save("fraud_transactions_map.html")

print("Map has been saved as 'fraud_transactions_map.html'")




# In[41]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from geopy.distance import geodesic
import folium
import matplotlib.pyplot as plt

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Feature Engineering
fraud_transactions_df['trans_date_trans_time'] = pd.to_datetime(fraud_transactions_df['trans_date_trans_time'])
fraud_transactions_df['transaction_hour'] = fraud_transactions_df['trans_date_trans_time'].dt.hour
fraud_transactions_df['transaction_day'] = fraud_transactions_df['trans_date_trans_time'].dt.dayofweek
fraud_transactions_df['transaction_month'] = fraud_transactions_df['trans_date_trans_time'].dt.month

# Calculate distance from home or usual location
def calculate_distance(row):
    return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km

fraud_transactions_df['distance_from_home'] = fraud_transactions_df.apply(calculate_distance, axis=1)

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long', 'transaction_hour', 'transaction_day', 'transaction_month', 'distance_from_home']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Define the Random Forest model and hyperparameters for Grid Search
model_rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train_smote, y_train_smote)

best_rf_model = grid_search.best_estimator_

# Predict and evaluate the model
y_pred_rf = best_rf_model.predict(X_test)
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

print("Best Random Forest Model:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation
from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(best_rf_model, X_imputed, y, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores for Random Forest:")
print(scores_rf)
print(f'Average F1 Score: {scores_rf.mean()}')

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Save the map as an HTML file
map_folium.save("fraud_transactions_map.html")

print("Map has been saved as 'fraud_transactions_map.html'")




# In[35]:


# Create a map centered around the average location of the transactions
map_center = [balanced_transactions['lat'].mean(), balanced_transactions['long'].mean()]
map_folium = folium.Map(location=map_center, zoom_start=12)

# Add fraudulent transactions to the map
for _, row in fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        popup=f'Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Add non-fraudulent transactions to the map
for _, row in non_fraud_transactions.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5,
        popup=f'Non-Fraudulent Transaction: {row["lat"]}, {row["long"]}'
    ).add_to(map_folium)

# Display the map in the Jupyter Notebook
display(map_folium)


# Detailed Explanation of the Results
# 1. Confusion Matrix and Classification Report:
# The confusion matrix and classification report provide a detailed breakdown of the model's performance in predicting fraudulent (1) and non-fraudulent (0) transactions.
# 
# Confusion Matrix:
# 
# True Positives (TP): 1,275 transactions were correctly identified as fraudulent.
# True Negatives (TN): 1,220 transactions were correctly identified as non-fraudulent.
# False Positives (FP): 282 transactions were incorrectly identified as fraudulent (when they were actually non-fraudulent).
# False Negatives (FN): 226 transactions were incorrectly identified as non-fraudulent (when they were actually fraudulent).
# Classification Metrics:
# 
# Precision for Class 0 (Non-Fraudulent): 0.84
# 
# Interpretation: Out of all the transactions predicted as non-fraudulent, 84% were actually non-fraudulent.
# Recall for Class 0: 0.81
# 
# Interpretation: The model correctly identified 81% of the actual non-fraudulent transactions.
# F1-Score for Class 0: 0.83
# 
# Interpretation: The harmonic mean of precision and recall for non-fraudulent transactions is 0.83.
# Precision for Class 1 (Fraudulent): 0.82
# 
# Interpretation: Out of all the transactions predicted as fraudulent, 82% were actually fraudulent.
# Recall for Class 1: 0.85
# 
# Interpretation: The model correctly identified 85% of the actual fraudulent transactions.
# F1-Score for Class 1: 0.83
# 
# Interpretation: The harmonic mean of precision and recall for fraudulent transactions is 0.83.
# Overall Accuracy: 0.83
# 
# Interpretation: The model correctly classified 83% of the transactions in the test set.
# Macro Average (Across Both Classes):
# 
# Precision, Recall, and F1-Score: All are 0.83
# Interpretation: The macro average gives equal weight to both classes, indicating that the model performs equally well across both fraudulent and non-fraudulent classes.
# Weighted Average:
# 
# Precision, Recall, and F1-Score: All are 0.83
# Interpretation: The weighted average accounts for the imbalance between classes, suggesting a balanced performance across the dataset.
# 2. Cross-Validation F1 Scores:
# The cross-validation F1 scores provide insight into the model's stability and generalization performance across different subsets of the data:
# 
# F1 Scores: [0.6722, 0.6381, 0.0861, 0.5575, 0.7322]
# 
# Interpretation: These scores indicate the F1 performance across five different folds during cross-validation. The F1 score is the harmonic mean of precision and recall, providing a balance between the two.
# Variability:
# 
# The F1 scores show significant variability across the folds, with one fold having a very low F1 score (0.0861). This suggests that the model's performance may be sensitive to the specific data in each fold, which could be due to factors like class imbalance or feature distributions that differ across folds.
# Average F1 Score: 0.5372
# 
# Interpretation: The average F1 score across all folds is 0.5372, indicating that the model has moderate performance when averaged across different subsets of the data.
# 3. What This Means:
# Model Strengths:
# 
# The model shows strong performance in identifying both fraudulent and non-fraudulent transactions, with high precision, recall, and F1-scores for both classes.
# The overall accuracy of 83% indicates that the model is reliable and has good generalization performance on the test set.
# Areas of Concern:
# 
# The significant variability in F1 scores during cross-validation suggests that the model's performance may not be consistent across different data splits. This could be due to a variety of factors such as:
# Data Imbalance: Even with SMOTE applied, there could be local imbalances or outliers in specific folds.
# Feature Sensitivity: The model might be over-relying on certain features that behave differently across folds.
# Comparison to Previous Model:
# 
# Compared to the previous model, which had an accuracy of 62% and a lower F1-score for both classes, the new model shows marked improvement:
# The F1-scores have improved from around 0.61-0.63 to 0.83, indicating better balance between precision and recall.
# The confusion matrix shows fewer misclassifications (lower FP and FN), suggesting that the model is better at distinguishing between fraudulent and non-fraudulent transactions.
# Conclusion:
# The Random Forest model with the updated feature set (including time-based features) and SMOTE for balancing the data has resulted in a robust fraud detection model. It performs well across both classes with an overall accuracy of 83%. However, the variability in cross-validation scores indicates potential instability, which might require further investigation, such as fine-tuning hyperparameters, exploring additional features, or employing more sophisticated cross-validation techniques.

# # Beyond SMOTE: 
# Handling Imbalanced Data
# Different Sampling Techniques: Beyond SMOTE, consider using SMOTE variants like Borderline-SMOTE or ADASYN. You can also explore ensemble methods that handle imbalanced data like 
# 
# 

# In[44]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import folium
from geopy.distance import geodesic

# Load the data
fraud_transactions_df = pd.read_csv("fraudtrain.csv")

# Feature Engineering
fraud_transactions_df['trans_date_trans_time'] = pd.to_datetime(fraud_transactions_df['trans_date_trans_time'])
fraud_transactions_df['transaction_hour'] = fraud_transactions_df['trans_date_trans_time'].dt.hour
fraud_transactions_df['transaction_day'] = fraud_transactions_df['trans_date_trans_time'].dt.dayofweek
fraud_transactions_df['transaction_month'] = fraud_transactions_df['trans_date_trans_time'].dt.month

# Calculate distance from home or usual location
def calculate_distance(row):
    return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km

fraud_transactions_df['distance_from_home'] = fraud_transactions_df.apply(calculate_distance, axis=1)

# Filter transactions for those that are fraudulent
fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 1]

# Create a balanced dataset by including non-fraud transactions
non_fraud_transactions = fraud_transactions_df[fraud_transactions_df['is_fraud'] == 0].sample(n=len(fraud_transactions), random_state=42)
balanced_transactions = pd.concat([fraud_transactions, non_fraud_transactions])

# Prepare features and target variable
X = balanced_transactions[['lat', 'long', 'merch_lat', 'merch_long', 'transaction_hour', 'transaction_day', 'transaction_month', 'distance_from_home']]
y = balanced_transactions['is_fraud']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Define models and techniques
models = {
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42, sampling_strategy='minority'),
    "SMOTE-ENN": SMOTEENN(random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(random_state=42)
}

# Results storage
results = {}

for name, technique in models.items():
    try:
        if name == "BalancedRandomForest":
            # Train Balanced Random Forest directly
            model = technique
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"\n{name}:")
            print(classification_report(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
            results[name] = model
        else:
            # Apply sampling technique
            X_train_resampled, y_train_resampled = technique.fit_resample(X_train, y_train)
            
            # Train Random Forest model
            rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            rf_model.fit(X_train_resampled, y_train_resampled)
            
            # Predict and evaluate
            y_pred = rf_model.predict(X_test)
            print(f"\nRandom Forest with {name}:")
            print(classification_report(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
            results[name] = rf_model

            # ROC Curve and AUC
            y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {name}')
            plt.legend(loc="lower right")
            plt.show()
    except ValueError as e:
        print(f"Error with {name}: {e}")

# Optional: Save the map if required


# 
# Model Comparison Explanation
# 1. Random Forest with Class Weights:
# 
# Precision: 0.63 (Class 0), 0.62 (Class 1)
# Recall: 0.60 (Class 0), 0.64 (Class 1)
# F1-Score: 0.61 (Class 0), 0.63 (Class 1)
# Accuracy: 0.62
# Confusion Matrix: [[906, 596], [540, 961]]
# Explanation: This model uses class weights to handle imbalanced data. While it performs reasonably well, it shows lower precision and recall compared to other models. The confusion matrix indicates a higher number of false positives and false negatives, reflecting the challenges in handling imbalanced data effectively.
# 
# 2. Random Forest with SMOTE:
# 
# Precision: 0.84 (Class 0), 0.82 (Class 1)
# Recall: 0.81 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0 and 1)
# Accuracy: 0.83
# Confusion Matrix: [[1220, 282], [226, 1275]]
# Explanation: SMOTE (Synthetic Minority Over-sampling Technique) improves the balance between classes by generating synthetic samples. This model shows significant improvement in precision, recall, and F1-score, indicating better handling of class imbalance and overall model performance.
# 
# 3. Random Forest with Borderline-SMOTE:
# 
# Precision: 0.85 (Class 0), 0.82 (Class 1)
# Recall: 0.81 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0), 0.84 (Class 1)
# Accuracy: 0.83
# Confusion Matrix: [[1221, 281], [222, 1279]]
# Explanation: Borderline-SMOTE focuses on creating samples near the decision boundary, which helps improve performance for harder-to-classify instances. The model achieves similar performance to SMOTE but with slightly better recall for Class 1, showing its effectiveness in capturing borderline cases.
# 
# 4. Random Forest with SMOTE-ENN:
# 
# Precision: 0.84 (Class 0), 0.78 (Class 1)
# Recall: 0.76 (Class 0), 0.86 (Class 1)
# F1-Score: 0.80 (Class 0), 0.82 (Class 1)
# Accuracy: 0.81
# Confusion Matrix: [[1148, 354], [214, 1287]]
# Explanation: SMOTE-ENN (SMOTE with Edited Nearest Neighbors) combines oversampling with cleaning techniques to remove noisy samples. This model shows a trade-off between precision and recall, with slightly lower accuracy but better recall for Class 1.
# 
# 5. BalancedRandomForest:
# 
# Precision: 0.85 (Class 0), 0.82 (Class 1)
# Recall: 0.82 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0), 0.84 (Class 1)
# Accuracy: 0.83
# Confusion Matrix: [[1227, 275], [223, 1278]]
# Explanation: BalancedRandomForestClassifier inherently handles class imbalance by balancing the classes within each tree. It shows strong performance with good precision, recall, and F1-scores for both classes, making it a robust choice for this dataset.
# 
# Conclusion
# By integrating advanced sampling techniques like SMOTE, Borderline-SMOTE, and SMOTE-ENN, and using the BalancedRandomForest, the model performance improved notably over the baseline Random Forest with class weights. SMOTE and Borderline-SMOTE provided balanced performance across both precision and recall, while SMOTE-ENN offered a good balance between precision and recall despite slightly lower accuracy. The BalancedRandomForest showed overall strong performance without additional sampling, demonstrating its effectiveness in handling imbalanced data directly.
# 
# Conclusion: The project successfully developed an advanced machine learning model for detecting financial fraud in gas station transactions. By incorporating time-based features, distance calculations, and geospatial analytics, the Random Forest classifier demonstrated enhanced accuracy and reliability. The application of SMOTE and other advanced sampling techniques, along with rigorous hyperparameter tuning, resulted in improved performance over earlier models. The final model achieved balanced precision and recall, highlighting its effectiveness in identifying fraudulent transactions while minimizing false positives and negatives.
# 
# Future Work
# Incorporate Additional Contextual Features: Adding features like transaction history or customer demographics could capture more detailed behavioral patterns and improve fraud detection.
# 
# Explore Deep Learning Techniques: Neural networks or other deep learning approaches may detect more subtle or complex fraud patterns, potentially improving model performance.
# 
# Real-Time Fraud Detection: Deploying the model in a live environment to flag suspicious transactions as they occur would enhance the fraud detection process in real-time.
# 
# Expand Geographic Scope: Integrating data from multiple financial institutions and expanding the geographic analysis could provide a broader view of fraud trends and improve detection accuracy.
# 
# Collaborate with Domain Experts: Working with experts in fraud prevention could refine feature engineering and ensure the model adapts to evolving fraud tactics.
# 
# Real-Time Fraud Detection
# Best Model for Real-Time Fraud Detection: The BalancedRandomForestClassifier and the Random Forest with SMOTE-based techniques are both strong candidates. BalancedRandomForestClassifier provides robust performance by inherently handling class imbalance, making it suitable for real-time applications. SMOTE and its variants are also effective but may require additional tuning for real-time scenarios. The choice of model will depend on the specific requirements of the deployment environment, including latency and computational resources.

# In[46]:


import pandas as pd

# Define the data
data = {
    "Model / Technique": [
        "Random Forest with Class Weights",
        "Random Forest with SMOTE",
        "Random Forest with Borderline-SMOTE",
        "Random Forest with SMOTE-ENN",
        "BalancedRandomForest"
    ],
    "Precision (Class 0)": [0.63, 0.84, 0.85, 0.84, 0.85],
    "Recall (Class 0)": [0.60, 0.81, 0.81, 0.76, 0.82],
    "F1-Score (Class 0)": [0.61, 0.83, 0.83, 0.80, 0.83],
    "Precision (Class 1)": [0.62, 0.82, 0.82, 0.78, 0.82],
    "Recall (Class 1)": [0.64, 0.85, 0.85, 0.86, 0.85],
    "F1-Score (Class 1)": [0.63, 0.83, 0.84, 0.82, 0.84],
    "Accuracy": [0.62, 0.83, 0.81, 0.81, 0.83],
    "Confusion Matrix": [
        "[[906, 596], [540, 961]]",
        "[[1220, 282], [226, 1275]]",
        "[[1221, 281], [222, 1279]]",
        "[[1148, 354], [214, 1287]]",
        "[[1227, 275], [223, 1278]]"
    ]
}

# Create DataFrame
df_comparison = pd.DataFrame(data)

# Display the DataFrame
df_comparison


# Model Comparison Explanation
# 1. Random Forest with Class Weights:
# 
# Precision: 0.63 (Class 0), 0.62 (Class 1)
# Recall: 0.60 (Class 0), 0.64 (Class 1)
# F1-Score: 0.61 (Class 0), 0.63 (Class 1)
# Accuracy: 0.62
# Confusion Matrix: [[906, 596], [540, 961]]
# Explanation: This model uses class weights to handle imbalanced data. While it performs reasonably well, it shows lower precision and recall compared to other models. The confusion matrix indicates a higher number of false positives and false negatives, reflecting the challenges in handling imbalanced data effectively.
# 
# 2. Random Forest with SMOTE:
# 
# Precision: 0.84 (Class 0), 0.82 (Class 1)
# Recall: 0.81 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0 and 1)
# Accuracy: 0.83
# Confusion Matrix: [[1220, 282], [226, 1275]]
# Explanation: SMOTE (Synthetic Minority Over-sampling Technique) improves the balance between classes by generating synthetic samples. This model shows significant improvement in precision, recall, and F1-score, indicating better handling of class imbalance and overall model performance.
# 
# 3. Random Forest with Borderline-SMOTE:
# 
# Precision: 0.85 (Class 0), 0.82 (Class 1)
# Recall: 0.81 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0), 0.84 (Class 1)
# Accuracy: 0.83
# Confusion Matrix: [[1221, 281], [222, 1279]]
# Explanation: Borderline-SMOTE focuses on creating samples near the decision boundary, which helps improve performance for harder-to-classify instances. The model achieves similar performance to SMOTE but with slightly better recall for Class 1, showing its effectiveness in capturing borderline cases.
# 
# 4. Random Forest with SMOTE-ENN:
# 
# Precision: 0.84 (Class 0), 0.78 (Class 1)
# Recall: 0.76 (Class 0), 0.86 (Class 1)
# F1-Score: 0.80 (Class 0), 0.82 (Class 1)
# Accuracy: 0.81
# Confusion Matrix: [[1148, 354], [214, 1287]]
# Explanation: SMOTE-ENN (SMOTE with Edited Nearest Neighbors) combines oversampling with cleaning techniques to remove noisy samples. This model shows a trade-off between precision and recall, with slightly lower accuracy but better recall for Class 1.
# 
# 5. BalancedRandomForest:
# 
# Precision: 0.85 (Class 0), 0.82 (Class 1)
# Recall: 0.82 (Class 0), 0.85 (Class 1)
# F1-Score: 0.83 (Class 0), 0.84 (Class 1)
# Accuracy: 0.83
# Confusion Matrix: [[1227, 275], [223, 1278]]
# Explanation: BalancedRandomForestClassifier inherently handles class imbalance by balancing the classes within each tree. It shows strong performance with good precision, recall, and F1-scores for both classes, making it a robust choice for this dataset.
# 
# Conclusion
# By integrating advanced sampling techniques like SMOTE, Borderline-SMOTE, and SMOTE-ENN, and using the BalancedRandomForest, the model performance improved notably over the baseline Random Forest with class weights. SMOTE and Borderline-SMOTE provided balanced performance across both precision and recall, while SMOTE-ENN offered a good balance between precision and recall despite slightly lower accuracy. The BalancedRandomForest showed overall strong performance without additional sampling, demonstrating its effectiveness in handling imbalanced data directly.
# 
# Conclusion: The project successfully developed an advanced machine learning model for detecting financial fraud in gas station transactions. By incorporating time-based features, distance calculations, and geospatial analytics, the Random Forest classifier demonstrated enhanced accuracy and reliability. The application of SMOTE and other advanced sampling techniques, along with rigorous hyperparameter tuning, resulted in improved performance over earlier models. The final model achieved balanced precision and recall, highlighting its effectiveness in identifying fraudulent transactions while minimizing false positives and negatives.
# 
# Future Work
# Incorporate Additional Contextual Features: Adding features like transaction history or customer demographics could capture more detailed behavioral patterns and improve fraud detection.
# 
# Explore Deep Learning Techniques: Neural networks or other deep learning approaches may detect more subtle or complex fraud patterns, potentially improving model performance.
# 
# Real-Time Fraud Detection: Deploying the model in a live environment to flag suspicious transactions as they occur would enhance the fraud detection process in real-time.
# 
# Expand Geographic Scope: Integrating data from multiple financial institutions and expanding the geographic analysis could provide a broader view of fraud trends and improve detection accuracy.
# 
# Collaborate with Domain Experts: Working with experts in fraud prevention could refine feature engineering and ensure the model adapts to evolving fraud tactics.
# 
# Real-Time Fraud Detection
# Best Model for Real-Time Fraud Detection: The BalancedRandomForestClassifier and the Random Forest with SMOTE-based techniques are both strong candidates. BalancedRandomForestClassifier provides robust performance by inherently handling class imbalance, making it suitable for real-time applications. SMOTE and its variants are also effective but may require additional tuning for real-time scenarios. The choice of model will depend on the specific requirements of the deployment environment, including latency and computational resources.

# In[ ]:




