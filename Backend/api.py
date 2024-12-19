import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve database credentials from environment variables
host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

def get_connection():
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

def add_worker(name):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO worker (name) VALUES (%s)", (name,))
    connection.commit()
    cursor.close()
    connection.close()

def get_worker(id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM worker WHERE id = %s", (id,))
    worker = cursor.fetchall()
    cursor.close()
    connection.close()
    return worker

# Check if the connection is successful
if __name__ == '__main__':
    connection = get_connection()
    if connection.is_connected():
        print("Successfully connected to the database")
    connection.close()