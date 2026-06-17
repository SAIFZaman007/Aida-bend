"""HTTP routes: projects + conversations (full CRUD) + streaming chat.

v2 changes:
 • Proper SSE: explicit `event:` lines (token / sentence / done / error) in
   addition to the `data:` payload, plus an `id:` per frame and a heartbeat
   comment to keep proxies/browsers from buffering or timing out — this is
   the "implementation of proper SSE" requested.
 • Frames now carry sentence-boundary events (see chat/service.py) so the
   frontend can drive the real-time reading-progress highlighter in sync
   with avatar speech.
 • Conversation/project deletes also invalidate the Redis history cache.
"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat import schemas
from src.chat.service import stream_reply
from src.core import redis_client
from src.core.database import get_session
from src.memory.models import Conversation, MemoryChunk, Message, Project

router = APIRouter()


# ---------------- Projects ----------------
@router.post("/projects", response_model=schemas.ProjectOut)
async def create_project(
    body: schemas.ProjectCreate, session: AsyncSession = Depends(get_session)
):
    project = Project(name=body.name, description=body.description)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


@router.get("/projects", response_model=list[schemas.ProjectOut])
async def list_projects(session: AsyncSession = Depends(get_session)):
    rows = await session.execute(select(Project).order_by(Project.created_at.desc()))
    return list(rows.scalars().all())


@router.patch("/projects/{project_id}", response_model=schemas.ProjectOut)
async def update_project(
    project_id: str,
    body: schemas.ProjectUpdate,
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    await session.commit()
    await session.refresh(project)
    return project


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, session: AsyncSession = Depends(get_session)):
    conv_ids = (
        await session.execute(
            select(Conversation.id).where(Conversation.project_id == project_id)
        )
    ).scalars().all()
    if conv_ids:
        await session.execute(delete(Message).where(Message.conversation_id.in_(conv_ids)))
        for cid in conv_ids:
            await redis_client.cache_invalidate(cid)
    await session.execute(delete(Conversation).where(Conversation.project_id == project_id))
    await session.execute(delete(MemoryChunk).where(MemoryChunk.project_id == project_id))
    await session.execute(delete(Project).where(Project.id == project_id))
    await session.commit()
    return {"deleted": project_id}


# ---------------- Conversations ----------------
@router.post("/conversations", response_model=schemas.ConversationOut)
async def create_conversation(
    body: schemas.ConversationCreate, session: AsyncSession = Depends(get_session)
):
    if await session.get(Project, body.project_id) is None:
        raise HTTPException(404, "Project not found")
    conv = Conversation(
        project_id=body.project_id, title=body.title, persona=body.persona
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


@router.get(
    "/projects/{project_id}/conversations",
    response_model=list[schemas.ConversationOut],
)
async def list_conversations(
    project_id: str, session: AsyncSession = Depends(get_session)
):
    rows = await session.execute(
        select(Conversation)
        .where(Conversation.project_id == project_id)
        .order_by(Conversation.created_at.desc())
    )
    return list(rows.scalars().all())


@router.patch("/conversations/{conversation_id}", response_model=schemas.ConversationOut)
async def update_conversation(
    conversation_id: str,
    body: schemas.ConversationUpdate,
    session: AsyncSession = Depends(get_session),
):
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    if body.title is not None:
        conv.title = body.title
    if body.persona is not None:
        conv.persona = body.persona
    await session.commit()
    await session.refresh(conv)
    return conv


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, session: AsyncSession = Depends(get_session)
):
    await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
    await session.commit()
    await redis_client.cache_invalidate(conversation_id)
    return {"deleted": conversation_id}


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[schemas.MessageOut],
)
async def list_messages(
    conversation_id: str, session: AsyncSession = Depends(get_session)
):
    rows = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list(rows.scalars().all())


# ---------------- Chat (streaming, proper SSE) ----------------
def _sse(event: str, frame_id: int, payload: dict) -> str:
    return f"id: {frame_id}\nevent: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat")
async def chat(body: schemas.ChatRequest, session: AsyncSession = Depends(get_session)):
    async def event_stream():
        frame_id = 0
        try:
            async for evt in stream_reply(
                session, body.conversation_id, body.content, body.remember
            ):
                frame_id += 1
                if evt["type"] == "token":
                    yield _sse("token", frame_id, {"t": evt["t"]})
                elif evt["type"] == "sentence":
                    yield _sse(
                        "sentence",
                        frame_id,
                        {"index": evt["index"], "text": evt["text"]},
                    )
        except Exception as exc:
            frame_id += 1
            yield _sse("error", frame_id, {"error": str(exc)})
        frame_id += 1
        yield _sse("done", frame_id, {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
