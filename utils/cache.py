from sqlalchemy.orm import Session

def evict_old_versions(model_name: str, current_version: str, db: Session):
    from db import get_db_session  # Or pass db directly

    db.execute("""
        DELETE FROM predictions
        WHERE model_name = :model_name AND model_version != :current_version
    """, {"model_name": model_name, "current_version": current_version})
    db.commit()