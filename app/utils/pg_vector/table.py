from datetime import datetime
from typing import Literal, Optional

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

JobStatusType = Literal[
    "completed",
    "pending",
    "processing",
    "failed",
    "canceled",
]


def is_job_terminated(status: JobStatusType) -> bool:
    return status in ["completed", "failed", "canceled"]


class OCRTable(Base):
    __tablename__ = "KnowledgeBaseOCR"

    id: str = Column(String, primary_key=True, nullable=False)
    url: str = Column(String, nullable=False)
    markdownUrl: Optional[str] = Column(String, nullable=True)
    jsonUrl: Optional[str] = Column(String, nullable=True)
    status: Optional[JobStatusType] = Column(String, nullable=True)
    createdAt: Optional[datetime] = Column(DateTime, nullable=True)
    updatedAt: Optional[datetime] = Column(DateTime, nullable=True)
    createdBy: Optional[str] = Column(String, nullable=True)
    updatedBy: Optional[str] = Column(String, nullable=True)
    tokenUsage: Optional[dict[str, dict[str, int]]] = Column(JSON, nullable=True)

    def __iter__(self):
        for column in self.__table__.columns:
            yield column.name, getattr(self, column.name)
