import pandas as pd

data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

s='python'

print(s[0:-1])
print(df.columns)

print(df[0:-1])
print[df['A']]

# Accessing the first column using index position
column_0 = df[df.columns[0]]


print(column_0)

print()





# import pandas as pd

# data = {'A': [1, 2, 3, 4],
#         'B': [5, 6, 7, 8]}

# # Creating a DataFrame with custom row labels
# #df = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4'])

# df=pd.DataFrame(data)

# print(df)

# # Using .loc to select a single row by label
# single_row_loc = df.loc[1]  # Selecting 'row2' by label

# # Using .iloc to select a single row by integer position
# single_row_iloc = df.iloc[1]  # Selecting the second row (position 1) by integer position

# print("Using .loc:")
# print(single_row_loc)

# print("\nUsing .iloc:")
# print(single_row_iloc)








# import pandas as pd
# from sqlalchemy import create_engine

# # Database connection parameters
# db_params = {
#     'database': 'ETL_project',
#     'user': 'postgres',
#     'password': 'postgres',
#     'host': 'localhost',
#     'port': '5432'
# }

# # CSV file path
# csv_file_path = 'C:/Users/ADMIN/Desktop/data_pipeline/diabetes.csv'

# # PostgreSQL table name
# table_name = 'diabetes_src'

# try:
#     # Create a connection to the PostgreSQL database using SQLAlchemy
#     engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')

#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file_path)

#     # Write the DataFrame to the PostgreSQL table
#     df.to_sql(table_name, engine, if_exists='replace', index=False)

#     print(f"CSV data from '{csv_file_path}' has been sent to '{table_name}' in the PostgreSQL database successfully.")

# except Exception as e:
#     print(f"Error: {e}")
