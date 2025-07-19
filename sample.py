from db.database import Base, engine
from db import models

def init():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized.")

if __name__ == "__main__":
    init()
