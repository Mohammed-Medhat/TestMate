from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'testmate.db')
_DB_URL = f"sqlite:///{os.path.abspath(_DB_PATH)}"

engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ExtractionRecord(Base):
    __tablename__ = "extraction_records"
    id = Column(Integer, primary_key=True, index=True)
    repo_name = Column(String, nullable=True)
    project_name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class UserFeedback(Base):
    __tablename__ = "user_feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    extraction_id = Column(Integer, nullable=True)
    repo_name = Column(String, nullable=True)
    feature_name = Column(String, nullable=True)
    original_value = Column(Text, nullable=True)
    refined_value = Column(Text, nullable=True)
    action = Column(String, nullable=True)
    user_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
