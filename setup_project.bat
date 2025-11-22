@echo off
echo ============================================
echo  Tiny SQL Expert - Project Setup (Windows)
echo ============================================

REM Create folders
mkdir models
mkdir tests
mkdir logs

REM Create schema.sql
echo PRAGMA foreign_keys = ON;> schema.sql
echo.>> schema.sql
echo CREATE TABLE users (^>^> schema.sql
echo     user_id INTEGER PRIMARY KEY,>> schema.sql
echo     name TEXT NOT NULL,>> schema.sql
echo     email TEXT UNIQUE NOT NULL,>> schema.sql
echo     signup_date TEXT>> schema.sql
echo );>> schema.sql
echo.>> schema.sql
echo CREATE TABLE products (^>^> schema.sql
echo     product_id INTEGER PRIMARY KEY,>> schema.sql
echo     name TEXT NOT NULL,>> schema.sql
echo     category TEXT,>> schema.sql
echo     price REAL NOT NULL>> schema.sql
echo );>> schema.sql
echo.>> schema.sql
echo CREATE TABLE orders (^>^> schema.sql
echo     order_id INTEGER PRIMARY KEY,>> schema.sql
echo     user_id INTEGER NOT NULL,>> schema.sql
echo     product_id INTEGER NOT NULL,>> schema.sql
echo     quantity INTEGER NOT NULL DEFAULT 1,>> schema.sql
echo     order_date TEXT,>> schema.sql
echo     status TEXT,>> schema.sql
echo     FOREIGN KEY(user_id) REFERENCES users(user_id),>> schema.sql
echo     FOREIGN KEY(product_id) REFERENCES products(product_id)>> schema.sql
echo );>> schema.sql

REM Create requirements.txt
echo llama-cpp-python>=0.1.49> requirements.txt
echo sqlparse>=0.4.4>> requirements.txt
echo tqdm>=4.65.0>> requirements.txt

REM Create placeholder tiny_sql_expert.py
echo # Placeholder script – replace with full version>> tiny_sql_expert.py
echo print("Project setup complete. Next: paste full tiny_sql_expert.py code.")>> tiny_sql_expert.py

REM Create README.md
echo # Tiny SQL Expert Project> README.md
echo Project initialized. Paste your Python script next.>> README.md

echo ============================================
echo  Setup Complete!
echo  Now you MUST paste the full Python script into
echo  tiny_sql_expert.py before running.
echo ============================================

pause
