import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus

# MongoDB connection setup
username = "cherukumallinithin2004"
password = "baddy"
encoded_password = quote_plus(password)

uri = f"mongodb+srv://{username}:{encoded_password}@cluster0.jkanm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

# Load your CSV file
data = pd.read_csv("/Users/nitin/Desktop/supply_chain_optimization/FMCG_data_augmented.csv")

# Convert DataFrame to a list of dictionaries
data_dict = data.to_dict("records")

# Insert data into the database
db = client["supply_chain_db"]
collection = db["fmcg_data"]

try:
    collection.insert_many(data_dict)
    print("Data successfully inserted into MongoDB!")
except Exception as e:
    print(f"An error occurred: {e}")

# Count the number of documents in the collection
count_in_mongo = collection.count_documents({})
print(f"Number of documents in MongoDB: {count_in_mongo}")

# Compare with the number of rows in the DataFrame
count_in_csv = len(data)
print(f"Number of rows in the original dataset: {count_in_csv}")

# Check if they match
if count_in_mongo == count_in_csv:
    print("All rows have been successfully inserted!")
else:
    print("There is a mismatch. Some rows might not have been inserted.")
