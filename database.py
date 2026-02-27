import sqlite3

Db_name = "screener.db"

def get_db_connection():
    connection = sqlite3.connect(Db_name)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_name TEXT NOT NULL,
        decision TEXT,
        authenticity TEXT,
        final_score REAL,
        skill_score REAL,
        exp_years TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
    )

    connection.commit()
    connection.close()

if __name__=="__main__":
    init_db()
    print("✅ Database initialized: screener.db")