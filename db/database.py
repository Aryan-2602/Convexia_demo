import sqlite3
import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import text


DB_PATH = os.path.join(os.getcwd(), "toxicity_predictions.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            id TEXT PRIMARY KEY,
            smiles TEXT NOT NULL,
            canonical_smiles TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            compound_id TEXT,
            model_name TEXT,
            model_version TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (compound_id) REFERENCES compounds(id)
        )
    """)

    conn.commit()
    conn.close()
    
def delete_old_predictions():
    db = SessionLocal()
    cutoff = datetime.now(timezone.UTC) - timedelta(days=90)
    db.execute(text("DELETE FROM predictions WHERE created_at < :cutoff"), {"cutoff": cutoff})
    db.commit()
    db.close()
