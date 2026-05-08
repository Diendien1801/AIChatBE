"""MongoDB chat persistence via LangChain MongoDBChatMessageHistory (SessionId / History fields)."""
from __future__ import annotations

import inspect
import os
from typing import List, Dict, Any
from urllib.parse import quote_plus, unquote

from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage

MONGO_DATABASE_NAME = "AICHAT1"
MONGO_COLLECTION_NAME = "chat_histories"


def normalize_mongo_uri(uri: str) -> str:
    """
    Escape username/password per RFC 3986 so PyMongo accepts URIs whose password
    contains @, :, /, etc. Uses the last '@' before the host (Atlas hostnames).
    """
    uri = uri.strip()
    for prefix in ("mongodb+srv://", "mongodb://"):
        if not uri.startswith(prefix):
            continue
        tail = uri[len(prefix) :]
        at = tail.rfind("@")
        if at == -1:
            return uri
        userinfo, host_and_more = tail[:at], tail[at + 1 :]
        if ":" not in userinfo:
            return uri
        user, password = userinfo.split(":", 1)
        user = unquote(user)
        password = unquote(password)
        new_userinfo = f"{quote_plus(user)}:{quote_plus(password)}"
        return f"{prefix}{new_userinfo}@{host_and_more}"
    return uri


def build_mongo_history(session_id: str) -> MongoDBChatMessageHistory:
    uri = os.getenv("MONGO_URI")
    if not uri or not uri.strip():
        raise ValueError("MONGO_URI is not configured")

    uri = normalize_mongo_uri(uri)

    kwargs: dict = {
        "connection_string": uri,
        "session_id": session_id,
        "database_name": MONGO_DATABASE_NAME,
        "collection_name": MONGO_COLLECTION_NAME,
    }

    params = inspect.signature(MongoDBChatMessageHistory.__init__).parameters
    if "session_id_key" in params:
        kwargs["session_id_key"] = "SessionId"
    if "history_key" in params:
        kwargs["history_key"] = "History"

    return MongoDBChatMessageHistory(**kwargs)


def messages_to_history_text(messages: List[BaseMessage]) -> str:
    if not messages:
        return ""

    lines = ["LỊCH SỬ TRÒ CHUYỆN:\n"]
    for m in messages:
        if m.type == "human":
            role_name = "USER"
        elif m.type == "ai":
            role_name = "BOT"
        else:
            role_name = m.type.upper()

        content = m.content
        if not isinstance(content, str):
            content = str(content)

        lines.append(f"- {role_name}: {content}\n")

    lines.append("\n")
    return "".join(lines)


def messages_to_chat_payload(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Serialize LangChain messages for JSON API (roles aligned with ChatMessage)."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.type == "human":
            role = "user"
        elif m.type == "ai":
            role = "bot"
        else:
            role = m.type

        content = m.content
        if not isinstance(content, str):
            content = str(content)

        out.append({"role": role, "content": content})
    return out
