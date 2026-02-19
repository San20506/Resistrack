"""HL7 v2.x message parser for ResisTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

FIELD_SEPARATOR: Final[str] = "|"
COMPONENT_SEPARATOR: Final[str] = "^"
SEGMENT_SEPARATOR: Final[str] = "\r"


@dataclass
class HL7Segment:
    """A single HL7 segment (e.g., MSH, PID, OBX)."""

    segment_id: str
    fields: list[str]

    def get_field(self, index: int) -> str:
        """Get field by 1-based index (HL7 convention)."""
        if index < 1 or index > len(self.fields):
            return ""
        return self.fields[index - 1]

    def get_component(self, field_index: int, component_index: int) -> str:
        """Get component from a field (1-based indices)."""
        field_value = self.get_field(field_index)
        components = field_value.split(COMPONENT_SEPARATOR)
        if component_index < 1 or component_index > len(components):
            return ""
        return components[component_index - 1]


@dataclass
class HL7Message:
    """Parsed HL7 v2.x message."""

    raw: str
    segments: list[HL7Segment] = field(default_factory=list)
    message_type: str = ""
    trigger_event: str = ""

    def get_segment(self, segment_id: str) -> HL7Segment | None:
        """Get first segment matching the ID."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        return None

    def get_all_segments(self, segment_id: str) -> list[HL7Segment]:
        """Get all segments matching the ID."""
        return [s for s in self.segments if s.segment_id == segment_id]


def parse_hl7_message(raw_message: str) -> HL7Message:
    """Parse a raw HL7 v2.x message string into structured data.

    Args:
        raw_message: Raw HL7 message with segment separators.

    Returns:
        Parsed HL7Message with segments, message type, and trigger event.

    Raises:
        ValueError: If message is empty or missing MSH segment.
    """
    if not raw_message or not raw_message.strip():
        msg = "HL7 message cannot be empty"
        raise ValueError(msg)

    # Normalize line endings
    normalized = raw_message.replace("\n", SEGMENT_SEPARATOR).replace("\r\r", "\r")
    segment_lines = [line.strip() for line in normalized.split(SEGMENT_SEPARATOR) if line.strip()]

    if not segment_lines:
        msg = "No segments found in HL7 message"
        raise ValueError(msg)

    message = HL7Message(raw=raw_message)

    for line in segment_lines:
        parts = line.split(FIELD_SEPARATOR)
        segment_id = parts[0]
        fields = parts[1:] if len(parts) > 1 else []
        message.segments.append(HL7Segment(segment_id=segment_id, fields=fields))

    # Extract message type from MSH segment
    msh = message.get_segment("MSH")
    if msh is None:
        msg = "HL7 message missing MSH segment"
        raise ValueError(msg)

    # MSH-9 contains message type (field 8 in 0-based, but MSH field numbering
    # starts at MSH-1=field separator, so MSH-9 = fields[7] after split)
    msg_type_field = msh.get_field(8)
    type_parts = msg_type_field.split(COMPONENT_SEPARATOR)
    message.message_type = type_parts[0] if len(type_parts) > 0 else ""
    message.trigger_event = type_parts[1] if len(type_parts) > 1 else ""

    return message
