from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


# Get password and username from environment variables
# pg_password = os.environ['PGPASS']
# pg_user = os.environ['PGUSER']
pg_password ='postgres'
pg_user = 'postgres'

# PostgreSQL database details
pg_host = 'localhost'
pg_port = 5432
pg_database = 'ETL_project'

# SQL Server database details
sql_server_driver = "ODBC Driver 17 for SQL Server"
sql_server_server = "192.168.29.128\SQLEXPRESS"
sql_server_database = "ETL_project"
sql_server_uid = "Krishna"
sql_server_pwd = "Kreshna@555"

# Extract data from PostgreSQL
def extract():
    try:
        pg_conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        pg_cursor = pg_conn.cursor()
        
        # Define the PostgreSQL tables to extract
        tables_to_extract = ['diabetes_src']  # Replace with your table names
        
        for table_name in tables_to_extract:
            df = pd.read_sql_query(f'SELECT * FROM {table_name}', pg_conn)
            load(df, table_name)
    except Exception as e:
        print("Data extract error: " + str(e))
    finally:
        pg_conn.close()

# Load data into SQL Server
def load(df, table_name):
    try:
        sql_server_conn_str = f"DRIVER={sql_server_driver};SERVER={sql_server_server};DATABASE={sql_server_database};UID={sql_server_uid};PWD={sql_server_pwd}"
        sql_server_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={sql_server_conn_str}")        
        print(f'Importing data into SQL Server table: {table_name}')
        df.to_sql(table_name, sql_server_engine, if_exists='replace', index=False)        
        print(f'Data imported successfully into SQL Server table: {table_name}')
        # For example, you can retrieve the data into a DataFrame
        
    except Exception as e:
        print("Data load error: " + str(e))

def Predict_status():
    try:
        sql_server_conn_str = f"DRIVER={sql_server_driver};SERVER={sql_server_server};DATABASE={sql_server_database};UID={sql_server_uid};PWD={sql_server_pwd}"
        sql_server_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={sql_server_conn_str}")    
        table_name= 'diabetes_src'   
        df = pd.read_sql(f"SELECT * FROM {table_name}", sql_server_engine)         
        print(df.isna)
        print(df.info)
        print(df.head())
        df.columns = df.columns.str.strip()
        print(df.columns) 
        threshold = 6.5
        df['diabetes_status'] = (df['glyhb'] >= threshold).astype(int)

        # Split the data into features (X) and the target variable (y)
        selected_features = ['chol','stab.glu','hdl','ratio','glyhb','age','height','weight','bp.1s','bp.1d','bp.2s','bp.2d','waist','hip','time.ppn']
        X = df[selected_features]  # Features
        y = df['diabetes_status']  # Target variable

        # Split the data into training and test sets (e.g., 80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create an imputer to fill missing values with the mean
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Create and train a logistic regression model with imputed data
        model = LogisticRegression()
        model.fit(X_train_imputed, y_train)

        # #Input weight value dynamically from the user
        # weight_value = int(input('Enter your weight in pounds: '))

        # # Create a new data point with the user-provided weight value
        # new_data_point = {
        #     'chol': X_test_imputed[:, selected_features.index('chol')].mean(),
        #     'stab.glu': X_test_imputed[:, selected_features.index('stab.glu')].mean(),
        #     'hdl': X_test_imputed[:, selected_features.index('hdl')].mean(),
        #     'ratio': X_test_imputed[:, selected_features.index('ratio')].mean(),
        #     'glyhb': X_test_imputed[:, selected_features.index('glyhb')].mean(),
        #     'age': X_test_imputed[:, selected_features.index('age')].mean(),
        #     'height': X_test_imputed[:, selected_features.index('height')].mean(),
        #     'weight': weight_value,  # Use the user-provided weight value
        #     'bp.1s': X_test_imputed[:, selected_features.index('bp.1s')].mean(),
        #     'bp.1d': X_test_imputed[:, selected_features.index('bp.1d')].mean(),
        #     'bp.2s': X_test_imputed[:, selected_features.index('bp.2s')].mean(),
        #     'bp.2d': X_test_imputed[:, selected_features.index('bp.2d')].mean(),
        #     'waist': X_test_imputed[:, selected_features.index('waist')].mean(),
        #     'hip': X_test_imputed[:, selected_features.index('hip')].mean(),
        #     'time.ppn': X_test_imputed[:, selected_features.index('time.ppn')].mean()
        # }

        # # Convert the new data point into a DataFrame
        # new_data_df = pd.DataFrame([new_data_point])

        # # Select the specific columns for prediction
        # new_data_features = new_data_df[selected_features]

        # Predict diabetes status for the new data point
        # prediction = model.predict(121)

        # if prediction[0] == 1:
        #     print(f"For weight {weight_value}, the person is predicted to have diabetes.")
        # else:
        #     print(f"For weight {weight_value}, the person is predicted to not have diabetes.")
        # Assuming you want to predict diabetes for a specific weight value (e.g., 121)
        weight_value = int(input("Enter the weight in pounds:"))
        prediction = model.predict([[121] + [0] * (len(selected_features) - 1)])

        if prediction[0] == 1:
            print(f"For weight {weight_value}, the person is predicted to have diabetes.")
        else:
            print(f"For weight {weight_value}, the person is predicted to not have diabetes.")

        
    except Exception as e:
        print("Data load error: " + str(e))

try:
    # Call the extract function to retrieve data from PostgreSQL
    #extract()
    Predict_status()

except Exception as e:
    print("Error while extracting and loading data: " + str(e))






# # Evaluate the model
        # y_pred = model.predict(X_test_imputed)
        # accuracy = accuracy_score(y_test, y_pred)
        # confusion = confusion_matrix(y_test, y_pred)

        # print(f"Accuracy: {accuracy}")
        # print("Confusion Matrix:")
        # print(confusion)


        # # Create and train a logistic regression model
        # model = LogisticRegression()
        # model.fit(X_train, y_train)

        # # Evaluate the model
        # y_pred = model.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # confusion = confusion_matrix(y_test, y_pred)

        # print(f"Accuracy: {accuracy}")
        # print("Confusion Matrix:")
        # print(confusion)










