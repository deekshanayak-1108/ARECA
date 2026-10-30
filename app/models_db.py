from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.db import Base, engine


class ScanRecord(Base):
    """Stores inference scan history and advisory records."""
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    disease_detected = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    gradcam_filename = Column(String(255), nullable=True)
    severity_level = Column(String(50), nullable=True, default="Moderate")
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "disease_detected": self.disease_detected,
            "confidence": round(self.confidence * 100, 2),
            "gradcam_filename": self.gradcam_filename,
            "severity_level": self.severity_level,
            "recommendation": self.recommendation,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


def init_db():
    """Initializes database tables if connection is available."""
    if engine is not None:
        try:
            Base.metadata.create_all(bind=engine)
            print("[INFO] Database tables verified/created successfully.")
        except Exception as e:
            print(f"[WARN] Could not create database tables automatically: {e}")
