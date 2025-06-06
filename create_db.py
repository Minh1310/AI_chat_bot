import sqlite3
import pandas as pd

# Kết nối SQLite
conn = sqlite3.connect('products.db')
cursor = conn.cursor()

# Tạo bảng
cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        price INTEGER,
        color TEXT,
        stock INTEGER,
        description TEXT
    )
''')

# Nhập dữ liệu từ CSV
df = pd.read_csv('products.csv')
df.to_sql('products', conn, if_exists='replace', index=False)

conn.commit()
conn.close()