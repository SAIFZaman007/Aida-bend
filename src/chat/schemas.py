"""API contracts. Pydantic models keep the frontend/backend boundary explicit."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agents.prompts import PERSONAS
from src.core.config import settings


def _check_persona(v: str | None) -> str | None:
    if v is not None and v not in PERSONAS:
        raise ValueError(f"unknown persona; expected one of {sorted(PERSONAS)}")
    return v


# --- Projects ---
class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Conversations ---
class ConversationCreate(BaseModel):
    project_id: str = Field(max_length=64)
    title: str = Field(default="New conversation", max_length=300)
    persona: str = "executive"

    _persona = field_validator("persona")(_check_persona)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    persona: str | None = None

    _persona = field_validator("persona")(_check_persona)


class ConversationOut(BaseModel):
    id: str
    project_id: str
    title: str
    persona: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Messages / chat ---
class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    conversation_id: str = Field(max_length=64)
    # Cap protects the paid API from oversized/abusive prompts.
    content: str = Field(min_length=1, max_length=settings.max_chat_chars)
    remember: bool = True
    # The mode the person is looking at. Optional: when sent it wins over the stored
    # mode (the "change mode" call may not have landed yet) and is stored itself.
    persona: str | None = None

    _persona = field_validator("persona")(_check_persona)
