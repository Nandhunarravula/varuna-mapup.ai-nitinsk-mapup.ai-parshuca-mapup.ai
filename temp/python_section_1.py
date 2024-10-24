1.REVERSE LIST BY N ELEMENTS: 

def reverse_by_n(lst, n):
    result = []
    for i in range(0, len(lst), n):
        # Reverse the group manually by swapping elements
        group = lst[i:i+n]  # Get the current group
        reversed_group = []
        for j in range(len(group)):
            reversed_group.insert(0, group[j])  # Insert each element at the beginning of the new list
        result.extend(reversed_group)  # Append the reversed group to the result
    return result

# Test cases
print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4)) # Output: [40, 30, 20, 10, 70, 60, 50]


2. Lists & Dictionaries:

def group_by_length(strings):
    length_dict = {}
    
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []  # Create a new list for this length if it doesn't exist
        length_dict[length].append(string)  # Append the string to the correct length list
    
    # Sort the dictionary by keys (string lengths) and return
    return dict(sorted(length_dict.items()))

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))  
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}

3. Flatten a Nested Dictionary :

def flatten_dict(nested_dict, parent_key='', result=None):
    if result is None:
        result = {}
    
    for key, value in nested_dict.items():
        new_key = parent_key + '.' + key if parent_key else key  # Concatenate keys with dot separator
        
        if isinstance(value, dict):
            # If the value is a dictionary, recursively flatten it
            flatten_dict(value, new_key, result)
        
        elif isinstance(value, list):
            # If the value is a list, iterate over elements and handle each one
            for i, item in enumerate(value):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    # If list element is a dictionary, recursively flatten it
                    flatten_dict(item, list_key, result)
                else:
                    # Otherwise, directly add the item to the result
                    result[list_key] = item
        
        else:
            # If the value is neither a dict nor a list, directly add it to the result
            result[new_key] = value
    
    return result

# Test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)

# Output:
# {
#     "road.name": "Highway 1",
#     "road.length": 350,
#     "road.sections[0].id": 1,
#     "road.sections[0].condition.pavement": "good",
#     "road.sections[0].condition.traffic": "moderate"
# }

4.Generate Unique Permutations:

def unique_permutations(nums):
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])  # Add a copy of the current permutation to result
            return
        
        for i in range(len(nums)):
            # Skip used elements or duplicates (when current element is the same as the previous one and previous wasn't used)
            if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                continue
            
            # Mark the element as used
            used[i] = True
            path.append(nums[i])
            
            # Recursively build the permutation
            backtrack(path, used)
            
            # Backtrack: remove the last element and mark it as unused
            path.pop()
            used[i] = False

    nums.sort()  # Sort the list to handle duplicates
    result = []
    used = [False] * len(nums)  # To track used elements in the current path
    backtrack([], used)
    return result

# Test case
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)

# Output:
# [
#     [1, 1, 2],
#     [1, 2, 1],
#     [2, 1, 1]
# ]

5. Find All Dates in a Text:

import re

def find_all_dates(text):
    # Define regex pattern for the three date formats
    pattern = r'\b(\d{2}-\d{2}-\d{4})\b|\b(\d{2}/\d{2}/\d{4})\b|\b(\d{4}\.\d{2}\.\d{2})\b'
    
    # Use re.findall to match the pattern
    matches = re.findall(pattern, text)
    
    # re.findall returns tuples of matches, we'll flatten the result
    dates = [match for group in matches for match in group if match]
    
    return dates

# Test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)

# Output:
# ["23-08-1994", "08/23/1994", "1994.08.23"]

6.Decode Polyline, Convert to DataFrame with Distances:

import polyline
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000

    # Convert latitudes and longitudes from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences between the coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in meters
    distance = R * c
    return distance

# Function to decode polyline and compute distances
def decode_polyline_to_df(polyline_str):
    # Decode the polyline string to a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a Pandas DataFrame from the decoded coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Initialize the distance column with 0 for the first row
    df['distance'] = 0.0
    
    # Calculate the distance for each row compared to the previous row
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)

    return df

# Example usage
polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'  # Example encoded polyline string
df = decode_polyline_to_df(polyline_str)

print(df)

7. Matrix Rotation and Transformation:

import numpy as np

def rotate_matrix_90_clockwise(matrix):
    # Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    rotated = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated[j][n - i - 1] = matrix[i][j]
    
    return rotated

def replace_with_row_col_sum(rotated_matrix):
    n = len(rotated_matrix)
    # Create arrays to store row sums and column sums
    row_sums = [sum(row) for row in rotated_matrix]
    col_sums = [sum(rotated_matrix[i][j] for i in range(n)) for j in range(n)]
    
    # Create the final matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Sum of all elements in the row and column excluding the element itself
            final_matrix[i][j] = row_sums[i] + col_sums[j] - rotated_matrix[i][j]
    
    return final_matrix

def transform_matrix(matrix):
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    
    # Step 2: Replace each element with the sum of its row and column excluding itself
    final_matrix = replace_with_row_col_sum(rotated_matrix)
    
    return final_matrix

# Example usage
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

result = transform_matrix(matrix)
print(np.array(result))  # Use numpy to print it in matrix form for clarity

# Output:
# [[22 19 16],[23 20 17],8.[24 21 18]]

8. Time Check:

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to check time coverage for each id, id_2 pair
def check_time_coverage(df):
    # Create a list of days from 0 (Monday) to 6 (Sunday)
    days_of_week = set(range(7))  # {0, 1, 2, 3, 4, 5, 6}
    
    # Initialize a list to store boolean results
    results = []
    
    # Group the DataFrame by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])
    
    # Function to convert time to seconds since midnight
    def time_to_seconds(t):
        return t.hour * 3600 + t.minute * 60 + t.second
    
    # Iterate through each group
    for (id_val, id_2_val), group in grouped:
        # Set to track which days are covered
        covered_days = set()
        
        # Dictionary to store time coverage for each day of the week
        day_coverage = {day: np.zeros(86400) for day in days_of_week}  # Array of 0s for 24-hour period (in seconds)
        
        # Iterate through each row in the group
        for _, row in group.iterrows():
            start_day = row['startDay']  # start day (0 for Monday, 6 for Sunday)
            end_day = row['endDay']      # end day (0 for Monday, 6 for Sunday)
            start_time = datetime.strptime(row['startTime'], '%H:%M:%S').time()
            end_time = datetime.strptime(row['endTime'], '%H:%M:%S').time()
            
            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)
            
            # Handle cases where the time range spans multiple days
            if start_day == end_day:
                # Fill in the coverage for the same day
                day_coverage[start_day][start_seconds:end_seconds] = 1
                covered_days.add(start_day)
            else:
                # Handle multi-day span
                day_coverage[start_day][start_seconds:] = 1  # From start time to the end of the day
                day_coverage[end_day][:end_seconds] = 1      # From the beginning of the day to end time
                covered_days.add(start_day)
                covered_days.add(end_day)
        
        # Check if all 7 days are covered
        all_days_covered = days_of_week == covered_days
        
        # Check if each day's 24-hour period is fully covered
        full_day_coverage = all(np.all(day_coverage[day] == 1) for day in covered_days)
        
        # Append the result (True if all days and 24-hour periods are covered, otherwise False)
        results.append((id_val, id_2_val, all_days_covered and full_day_coverage))
    
    # Convert results into a MultiIndex Series
    result_df = pd.DataFrame(results, columns=['id', 'id_2', 'valid'])
    result_series = result_df.set_index(['id', 'id_2'])['valid']
    
    return result_series

# Example usage
# Assume 'dataset-1.csv' is read into a DataFrame
df = pd.read_csv('dataset-1.csv')

# Apply the function
result_series = check_time_coverage(df)
print(result_series)



