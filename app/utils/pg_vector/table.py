from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, DateTime, String
from typing import Optional, Literal
from datetime import datetime

Base = declarative_base()

status_type = Literal[
    "completed",
    "pending",
    "retrying",
    "processing",
    "failed",
    "canceled",
]

class OCRTable(Base):
    __tablename__ = "KnowledgeBaseOCR"

    id: str = Column(String, primary_key=True, nullable=False)
    url: str = Column(String, nullable=False)
    markdownUrl: Optional[str] = Column(String, nullable=True)
    jsonUrl: Optional[str] = Column(String, nullable=True)
    status: Optional[status_type] = Column(String, nullable=True)
    createdAt: Optional[datetime] = Column(DateTime, nullable=True)
    updatedAt: Optional[datetime] = Column(DateTime, nullable=True)
    createdBy: Optional[str] = Column(String, nullable=True)
    updatedBy: Optional[str] = Column(String, nullable=True)

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)