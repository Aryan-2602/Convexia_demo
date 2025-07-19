# db/models.py
import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from datetime import datetime
from db.database import Base

# Use PG_UUID if using Postgres, otherwise fallback to string
UUID_TYPE = PG_UUID(as_uuid=True) if PG_UUID else String

class Compound(Base):
    __tablename__ = "compounds"

    id = Column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    smiles = Column(Text, nullable=False)
    canonical_smiles = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    compound_id = Column(UUID_TYPE, ForeignKey("compounds.id"))
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    prediction = Column(JSON, nullable=False)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
