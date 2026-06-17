"""API contracts. Pydantic models keep the frontend/backend boundary explicit."""
from datetime import datetime

from pydantic import BaseModel, Field


# --- Projects ---
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Conversations ---
class ConversationCreate(BaseModel):
    project_id: str
    title: str = "New conversation"
    persona: str = "executive"


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    persona: str | None = None


class ConversationOut(BaseModel):
    id: str
    project_id: str
    title: str
    persona: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Messages / chat ---
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    conversation_id: str
    content: str = Field(min_length=1)
    remember: bool = True