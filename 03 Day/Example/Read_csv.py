import pandas as pd

# Create the dataset
data = {
    'Age': [25, 30, 45, 22, 35, 29, 41, 50, 27, 39, 31, 28, 42, 26, 33],
    'Department': ['Sales', 'Engineering', 'HR', 'Sales', 'HR', 'Engineering', 'Sales', 'HR', 'Engineering', 'Sales', 'HR', 'Sales', 'Engineering', 'HR', 'Sales'],
    'Education': ['Bachelor', 'Master', 'High School', 'High School', 'Bachelor', 'Master', 'Bachelor', 'High School', 'Master', 'Bachelor', 'Master', 'High School', 'Bachelor', 'Master', 'High School'],
    'Years_at_Company': [1, 3, 10, 2, 7, 4, 5, 20, 6, 3, 8, 2, 9, 1, 4],
    'Job_Satisfaction': [3, 4, 2, 4, 3, 3, 1, 2, 4, 3, 4, 2, 3, 1, 4],
    'Attrition': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV file
csv_file_path = 'p4n_employee.csv'
df.to_csv(csv_file_path, index=False)
