#!/usr/bin/env python
# coding: utf-8

#Question 9: Distance Matrix Calculation
# Create a function named calculate_distance_matrix that takes the dataset-2.csv as input and generates a DataFrame representing distances between IDs.
# 
# The resulting DataFrame should have cumulative distances along known routes, with diagonal values set to 0. If distances between toll locations A to B and B to C are known, then the distance from A to C should be the sum of these distances. Ensure the matrix is symmetric, accounting for bidirectional distances between toll locations (i.e. A to B is equal to B to A).

# In[1]:


import pandas as pd
url = 'https://raw.githubusercontent.com/mapup/MapUp-DA-Assessment-2024/refs/heads/main/datasets/dataset-1.csv'

df = pd.read_csv(url)
df.to_csv('local_dataset.csv', index=False)
print("CSV file downloaded and saved as 'local_dataset.csv'.")


# In[2]:


import pandas as pd

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print("Columns in the DataFrame:", df.columns.tolist())
    print("Data types of columns:", df.dtypes)

    df.columns = df.columns.str.strip()
    
    from_col ="from"
    to_col ="to"
    distance_col="distance"

    if 'from_col' not in df.columns or 'to_col' not in df.columns or 'distance_col' not in df.columns:
        raise KeyError(f"Ensure the CSV file contains '{from_col}', '{to_col}', and '{distance_col}' columns.")
    toll_ids = set(df['from'].unique()).union(set(df['to'].unique()))
    distance_matrix = pd.DataFrame(float('inf'), index=toll_ids, columns=toll_ids)
    
    for _, row in df.iterrows():
        distance_matrix.at[row['from_col'], row['to_col']] = row['distance_col']
        distance_matrix.at[row['to_col'], row['from_col']] = row['distance_col']  
    
    for id in toll_ids:
        distance_matrix.at[id, id] = 0

    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
            
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

file_path = r"C:\Users\HP\Downloads\local_dataset 2.csv"

try:
    distance_matrix = calculate_distance_matrix(file_path)
    print(distance_matrix)
except Exception as e:
    print("Error:", e)


# In[3]:


print(df.columns)


# In[7]:


print(df)


# In[8]:


df


#10.Unroll Distance Matrix
# Create a function unroll_distance_matrix that takes the DataFrame created in Question 9. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.
# 
# All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.

# In[15]:


import pandas as pd

df = pd.read_csv(r"C:\Users\HP\Downloads\local_dataset 2.csv")

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll the dataset into a DataFrame with 'id_start', 'id_end', and 'distance'.
    
    Args:
        df (pandas.DataFrame): The input dataset containing the distances between different IDs.

    Returns:
        pd.DataFrame: A DataFrame containing unrolled distance values with 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = df[['id', 'id_2', 'able2Hov2']].copy()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    unrolled_df = unrolled_df[unrolled_df['id_end'] != -1]
    
    return unrolled_df

unrolled_result = unroll_distance_matrix(df)

unrolled_result.head()


# ### Question 11: Finding IDs within Percentage Threshold
# Create a function find_ids_within_ten_percentage_threshold that takes the DataFrame created in Question 10 and a reference value from the id_start column as an integer.
# 
# Calculate average distance for the reference value given as an input and return a sorted list of values from id_start column which lie within 10% (including ceiling and floor) of the reference value's average.

# In[23]:


import pandas as pd

data = pd.DataFrame({
    'id_start': [4, 5, 6, 7, 8],
    'id_end': [1050000, 1050001, 1200000, 1200000, 1200000],
    'distance': [6.0, 6.0, None, 6.0, 6.0]  
})

print("Sample DataFrame:")
print(data)

def find_ids_within_ten_percentage_threshold(df, reference_id):
    if reference_id not in df['id_start'].values:
        raise ValueError("Reference ID not found in the DataFrame.")

    reference_row = df[df['id_start'] == reference_id]
    
    average_distance = reference_row['distance'].mean()
    
    
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    
    filtered_ids = df[(df['distance'] >= lower_bound) & 
                      (df['distance'] <= upper_bound) & 
                      df['distance'].notna()]
    

    return sorted(filtered_ids['id_start'].unique().tolist())

reference_id = 4
result = find_ids_within_ten_percentage_threshold(data, reference_id)
print(f"IDs within 10% of the average distance for reference ID {reference_id}: {result}")


# In[24]:


result


# 12: Calculate Toll Rate
# Create a function calculate_toll_rate that takes the DataFrame created in Question 10 as input and calculates toll rates based on vehicle types.
# 
# The resulting DataFrame should add 5 columns to the input DataFrame: moto, car, rv, bus, and truck with their respective rate coefficients. The toll rates should be calculated by multiplying the distance with the given rate coefficients for each vehicle type:
# 
# 0.8 for moto
# 1.2 for car
# 1.5 for rv
# 2.2 for bus
# 3.6 for truck

# In[25]:


import pandas as pd

# Creating the DataFrame based on the provided data
data = pd.DataFrame({
    'id_start': [4, 5, 6, 7, 8],
    'id_end': [1050000, 1050001, 1200000, 1200000, 1200000],
    'distance': [6.0, 6.0, None, 6.0, 6.0]  
})

print("Sample DataFrame:")
print(data)

def calculate_toll_rate(df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate  
        
    return df


result_df = calculate_toll_rate(data)

print("\nDataFrame with Toll Rates:")
print(result_df)


# In[26]:


result_df


# 13: Calculate Time-Based Toll Rates
# Create a function named calculate_time_based_toll_rates that takes the DataFrame created in Question 12 as input and calculates toll rates for different time intervals within a day.
# 
# The resulting DataFrame should have these five columns added to the input: start_day, start_time, end_day, and end_time.
# 
# start_day, end_day must be strings with day values (from Monday to Sunday in proper case)
# start_time and end_time must be of type datetime.time() with the values from time range given below.
# Modify the values of vehicle columns according to the following time ranges:
# 
# Weekdays (Monday - Friday):
# 
# From 00:00:00 to 10:00:00: Apply a discount factor of 0.8
# From 10:00:00 to 18:00:00: Apply a discount factor of 1.2
# From 18:00:00 to 23:59:59: Apply a discount factor of 0.8
# Weekends (Saturday and Sunday):
# 
# Apply a constant discount factor of 0.7 for all times.
# For each unique (id_start, id_end) pair, cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
# 
# 

# In[19]:


import pandas as pd
from datetime import time

data = pd.DataFrame({
    'id_start': [4, 5, 6, 7, 8],
    'id_end': [1050000, 1050001, 1200000, 1200000, 1200000],
    'distance': [6.0, 6.0, None, 6.0, 6.0], 
    'moto': [4.8, 4.8, None, 4.8, 4.8],
    'car': [7.2, 7.2, None, 7.2, 7.2],
    'rv': [9.0, 9.0, None, 9.0, 9.0],
    'bus': [13.2, 13.2, None, 13.2, 13.2],
    'truck': [21.6, 21.6, None, 21.6, 21.6]
})

def calculate_time_based_toll_rates(df):
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    start_days = []
    end_days = []
    start_times = []
    end_times = []

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        for day in days_of_week:
            for hour in range(24):
                start_time = time(hour, 0, 0) 
                end_time = time(hour, 59, 59)   

                start_days.append(day)
                end_days.append(day)
                start_times.append(start_time)
                end_times.append(end_time)
    expanded_df = pd.DataFrame({
        'start_day': start_days,
        'start_time': start_times,
        'end_day': end_days,
        'end_time': end_times
    })

    expanded_df = pd.concat([expanded_df, df], axis=1)
    def apply_discount(row):
        start_day = row['start_day']
        start_time = row['start_time']
        vehicle_rates = {
            'moto': row['moto'],
            'car': row['car'],
            'rv': row['rv'],
            'bus': row['bus'],
            'truck': row['truck']
        }

        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']: 
            if time(0, 0) <= start_time < time(10, 0):
                discount_factor = 0.8
            elif time(10, 0) <= start_time < time(18, 0):
                discount_factor = 1.2
            else:  
                discount_factor = 0.8
        else:  
            discount_factor = 0.7

        for vehicle in vehicle_rates.keys():
            if vehicle_rates[vehicle] is not None:  
                vehicle_rates[vehicle] *= discount_factor
        
        return pd.Series(vehicle_rates)
    discounted_rates = expanded_df.apply(apply_discount, axis=1)
    expanded_df[['moto', 'car', 'rv', 'bus', 'truck']] = discounted_rates

    return expanded_df

result_df = calculate_time_based_toll_rates(data)

print("\nDataFrame with Time-Based Toll Rates:")
print(result_df)


# In[22]:


result_df


# In[ ]:



