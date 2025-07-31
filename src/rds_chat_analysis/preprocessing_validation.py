"""
This module provides Pydantic validation for chat log preprocessing.

See notebooks/v2/01-prepare-data.ipynb for example usage.
"""

from typing import Any, Dict, List, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


def check_null_bytes(obj: Any, path: str = "") -> None:
    """Recursively check for null bytes in any string field"""
    if isinstance(obj, str):
        if "\x00" in obj:
            raise ValueError(f"null byte found in string at {path or 'root'}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and "\x00" in key:
                raise ValueError(f"null byte found in dict key at {path}.{key}")
            check_null_bytes(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            check_null_bytes(item, f"{path}[{i}]" if path else f"[{i}]")


class ProcessedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(..., description="Unique message ID")
    text: str = Field(..., description="Message content")
    metadata: dict[str, JsonValue] = Field(..., description="Message metadata (JSONB)")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("id must be a non-empty string")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("text must be a string")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")

        if "log_id" not in v or not isinstance(v["log_id"], str):
            raise ValueError("metadata must contain 'log_id'")
        if "role" not in v or not isinstance(v["role"], str):
            raise ValueError("metadata must contain 'role'")

        return v

    @model_validator(mode="after")
    def validate_no_null_bytes(self) -> Self:
        check_null_bytes(self.model_dump())
        return self


class ProcessedLogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    messages: List[ProcessedMessage]

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(
        cls, v: List[ProcessedMessage]
    ) -> List[ProcessedMessage]:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ProcessedLogOutput":
        ids = [msg.id for msg in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate message IDs found")
        return self


def validate_processed_log(
    processed_data: List[Dict[str, Any]],
) -> bool:
    """Validate output from process_*_log functions"""
    _ = ProcessedLogOutput.model_validate({"messages": processed_data})
    return True


def validate_processed_message(
    processed_data: Dict[str, Any],
) -> bool:
    _ = ProcessedMessage.model_validate(processed_data)
    return True
