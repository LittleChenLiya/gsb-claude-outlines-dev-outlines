"""Incremental validation of structured (JSON) model outputs.

Black-box models do not constrain generation with a logits processor, so the
output they return may not actually conform to the requested output type. This
module validates model output as it is streamed in and reports, on failure:

* ``verified_prefix_length``: how many characters at the start of the output
  are still a valid prefix of the expected structure;
* ``failure_position``: the character position at which the output became
  invalid (``None`` if the stream simply ended too early).

The key safety property enforced here is that an unverified suffix of the
output is never reported or yielded as a successful result: the moment the
accumulated output is no longer an extendable prefix of valid JSON, a
:class:`~outlines.exceptions.StructuredGenerationError` is raised.
"""

import json
from dataclasses import dataclass
from json.decoder import WHITESPACE  # type: ignore[attr-defined]
from typing import Any, Iterator, List, Optional

import jsonschema

from outlines.exceptions import StructuredGenerationError
from outlines.types import JsonSchema
from outlines.types.utils import is_native_dict

__all__ = [
    "StructuredOutputValidator",
    "build_validator",
    "needs_structured_validation",
    "validate_structured_content",
]


# ---------------------------------------------------------------------------
# Incremental JSON prefix parser
# ---------------------------------------------------------------------------


class _Truncated(Exception):
    """Raised internally when a JSON value ends because the input ends early.

    ``position`` is the index immediately after the longest verified prefix
    (i.e. ``len(text)`` at the point parsing ran out of input).
    """

    def __init__(self, position: int):
        self.position = position
        super().__init__(f"truncated at position {position}")


def _skip_ws(text: str, idx: int) -> int:
    match = WHITESPACE.match(text, idx)
    return match.end()


def _parse_string(text: str, idx: int) -> int:
    """Parse a JSON string starting at the opening quote.

    Returns the index just past the closing quote. Raises ``_Truncated`` if
    the string is still open when the input ends.
    """
    length = len(text)
    idx += 1  # consume opening quote
    while True:
        if idx >= length:
            raise _Truncated(length)
        char = text[idx]
        if ord(char) < 0x20:
            raise json.JSONDecodeError(
                "Invalid control character in string", text, idx
            )
        if char == '"':
            return idx + 1
        if char == "\\":
            idx += 1
            if idx >= length:
                raise _Truncated(length)
            escape = text[idx]
            if escape in '"\\/bfnrt':
                idx += 1
            elif escape == "u":
                # Exactly 4 hexadecimal digits must follow.
                digits_start = idx + 1
                digits_end = digits_start + 4
                if length < digits_end:
                    raise _Truncated(length)
                hexdigits = text[digits_start:digits_end]
                if any(ch not in "0123456789abcdefABCDEF" for ch in hexdigits):
                    raise json.JSONDecodeError(
                        "Invalid \\uXXXX escape", text, idx
                    )
                idx = digits_end
            else:
                raise json.JSONDecodeError(
                    f"Invalid escape: {escape!r}", text, idx
                )
        else:
            idx += 1


def _parse_value(text: str, idx: int) -> int:
    """Parse one JSON value starting at ``idx``; return the end index."""
    if idx >= len(text):
        raise _Truncated(len(text))
    char = text[idx]
    if char == '"':
        return _parse_string(text, idx)
    if char == "{":
        return _parse_object(text, idx)
    if char == "[":
        return _parse_array(text, idx)
    if char in "-0123456789":
        # Delegate number parsing to the stdlib scanner. It raises
        # StopIteration when the number is cut off mid-token and
        # JSONDecodeError for genuinely malformed numbers.
        try:
            scanner = json.JSONDecoder().scan_once  # type: ignore[attr-defined]
            _, end = scanner(text, idx)
            return end
        except StopIteration:
            raise _Truncated(len(text)) from None
    for literal in ("true", "false", "null"):
        if text.startswith(literal, idx):
            return idx + len(literal)
        if literal.startswith(text[idx:]):
            raise _Truncated(len(text))
    raise json.JSONDecodeError("Expecting value", text, idx)


def _parse_object(text: str, idx: int) -> int:
    length = len(text)
    idx += 1  # consume '{'
    idx = _skip_ws(text, idx)
    if idx >= length:
        raise _Truncated(length)
    if text[idx] == "}":
        return idx + 1
    while True:
        idx = _skip_ws(text, idx)
        if idx >= length:
            raise _Truncated(length)
        if text[idx] != '"':
            raise json.JSONDecodeError(
                "Expecting property name enclosed in double quotes", text, idx
            )
        idx = _parse_string(text, idx)
        idx = _skip_ws(text, idx)
        if idx >= length:
            raise _Truncated(length)
        if text[idx] != ":":
            raise json.JSONDecodeError("Expecting ':' delimiter", text, idx)
        idx += 1
        idx = _skip_ws(text, idx)
        idx = _parse_value(text, idx)
        idx = _skip_ws(text, idx)
        if idx >= length:
            raise _Truncated(length)
        if text[idx] == ",":
            idx += 1
            continue
        if text[idx] == "}":
            return idx + 1
        raise json.JSONDecodeError("Expecting ',' delimiter", text, idx)


def _parse_array(text: str, idx: int) -> int:
    length = len(text)
    idx += 1  # consume '['
    idx = _skip_ws(text, idx)
    if idx >= length:
        raise _Truncated(length)
    if text[idx] == "]":
        return idx + 1
    while True:
        idx = _skip_ws(text, idx)
        idx = _parse_value(text, idx)
        idx = _skip_ws(text, idx)
        if idx >= length:
            raise _Truncated(length)
        if text[idx] == ",":
            idx += 1
            continue
        if text[idx] == "]":
            return idx + 1
        raise json.JSONDecodeError("Expecting ',' delimiter", text, idx)


@dataclass(frozen=True)
class _PrefixStatus:
    """Result of classifying a string against the JSON grammar.

    Attributes
    ----------
    state
        ``"complete"`` if the text is a complete JSON document (optional
        trailing whitespace), ``"incomplete"`` if the text is still a valid
        extendable prefix of a JSON document, or ``"invalid"`` if no JSON
        document can start with this text.
    verified_prefix_length
        Longest prefix length that is guaranteed valid so far.
    failure_position
        For ``"invalid"`` text, the character position at which the output
        becomes irrecoverably wrong.
    value
        The parsed Python value when the state is ``"complete"``.
    error
        The underlying ``JSONDecodeError`` for ``"invalid"`` text, if any.
    """

    state: str
    verified_prefix_length: int
    failure_position: Optional[int] = None
    value: Any = None
    error: Optional[json.JSONDecodeError] = None


def _classify_json_prefix(text: str) -> _PrefixStatus:
    """Classify *text* as a complete, incomplete, or invalid JSON document."""
    length = len(text)
    start = _skip_ws(text, 0)
    if start >= length:
        return _PrefixStatus("incomplete", length)
    try:
        end = _parse_value(text, start)
    except _Truncated as exc:
        return _PrefixStatus("incomplete", min(exc.position, length))
    except json.JSONDecodeError as exc:
        return _PrefixStatus(
            "invalid",
            min(exc.pos, length),
            failure_position=exc.pos,
            error=exc,
        )
    tail_end = _skip_ws(text, end)
    if tail_end == length:
        value = json.loads(text)
        return _PrefixStatus("complete", length, value=value)
    # A complete JSON value followed by non-whitespace content is invalid.
    return _PrefixStatus(
        "invalid",
        end,
        failure_position=end,
        error=json.JSONDecodeError("Extra data", text, end),
    )


# ---------------------------------------------------------------------------
# Output-type detection
# ---------------------------------------------------------------------------


def needs_structured_validation(output_type: Any) -> bool:
    """Return whether the output type denotes JSON-structured output."""
    if output_type is None:
        return False
    if isinstance(output_type, JsonSchema):
        return True
    if is_native_dict(output_type):
        return True
    return JsonSchema.is_json_schema(output_type)


def _build_schema_validator(output_type):
    if is_native_dict(output_type):
        return None
    schema_dict = JsonSchema.convert_to(output_type, ["dict"])
    return jsonschema.Draft7Validator(schema_dict)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class StructuredOutputValidator:
    """Incrementally validate streamed output against a JSON output type.

    Raises :class:`StructuredGenerationError` as soon as the accumulated text
    is no longer a valid prefix of the expected JSON document, or at
    ``finish``/``validate_complete`` when the document is incomplete or fails
    the JSON schema.
    """

    def __init__(self, output_type: Any):
        if not needs_structured_validation(output_type):
            raise ValueError(
                "StructuredOutputValidator only supports JSON output types; "
                f"got {output_type!r}"
            )
        self.output_type = output_type
        self._schema_validator = _build_schema_validator(output_type)
        self._buffer = ""
        self._verified_length = 0
        self._failed = False

    @property
    def verified_prefix_length(self) -> int:
        """Length of the longest verified JSON prefix seen so far."""
        return self._verified_length

    def update(self, chunk: str) -> Iterator[str]:
        """Append a streamed chunk and yield newly verified text.

        Only characters belonging to the verified prefix are yielded. If the
        chunk makes the output invalid the unverified suffix is withheld and
        :class:`StructuredGenerationError` is raised.
        """
        if self._failed:
            raise StructuredGenerationError(
                "Structured generation already failed; the stream cannot "
                "recover from an invalid JSON prefix.",
                content=self._buffer + chunk,
                verified_prefix_length=self._verified_length,
                failure_position=self._verified_length,
                reason="update_after_failure",
            )
        self._buffer += chunk
        status = _classify_json_prefix(self._buffer)

        if status.state == "invalid":
            self._failed = True
            self._verified_length = status.verified_prefix_length
            raise self._build_syntax_error(self._buffer, status)

        # Everything up to the verified length is safe to hand out.
        safe_end = status.verified_prefix_length
        if safe_end > self._verified_length:
            new_text = self._buffer[self._verified_length:safe_end]
            self._verified_length = safe_end
            yield new_text

    def finish(self) -> str:
        """Mark the stream complete; return the full verified output.

        Raises if the output is truncated or fails schema validation.
        """
        return self.validate_complete(self._buffer)

    def validate_complete(self, text: str) -> str:
        """Validate a complete response and return it unchanged."""
        status = _classify_json_prefix(text)
        if status.state == "invalid":
            self._failed = True
            self._verified_length = status.verified_prefix_length
            raise self._build_syntax_error(text, status)
        if status.state == "incomplete":
            self._failed = True
            self._verified_length = status.verified_prefix_length
            raise StructuredGenerationError(
                "Structured generation failed: the model ended the output "
                "before producing a complete JSON document "
                f"(verified prefix length: {status.verified_prefix_length}, "
                "failure position: null - the output ended early).",
                content=text,
                verified_prefix_length=status.verified_prefix_length,
                failure_position=None,
                reason="incomplete",
            )
        # state == "complete": check the JSON schema, if any.
        self._check_schema(text, status.value)
        self._buffer = text
        self._verified_length = len(text)
        return text

    def _build_syntax_error(
        self, text: str, status: _PrefixStatus
    ) -> StructuredGenerationError:
        pos = status.failure_position
        if pos is None:  # pragma: no cover - always set for invalid
            pos = status.verified_prefix_length
        line = text.count("\n", 0, pos) + 1
        last_newline = text.rfind("\n", 0, pos)
        column = pos + 1 if last_newline == -1 else pos - last_newline
        snippet = text[max(0, pos - 10):pos + 10]
        detail = status.error.msg if status.error is not None else "Invalid JSON"
        message = (
            "Structured generation failed: invalid JSON at position "
            f"{pos} (line {line}, column {column}): {detail}. "
            f"Verified prefix length: {status.verified_prefix_length}. "
            f"Near: {snippet!r}."
        )
        return StructuredGenerationError(
            message,
            content=text,
            verified_prefix_length=status.verified_prefix_length,
            failure_position=pos,
            reason="invalid_json_syntax",
            original_exception=status.error,
        )

    def _check_schema(self, text: str, value: Any) -> None:
        if self._schema_validator is None:
            return
        errors: List[jsonschema.ValidationError] = sorted(
            self._schema_validator.iter_errors(value),
            key=lambda err: list(err.absolute_path),
        )
        if not errors:
            return
        self._failed = True
        error = errors[0]
        location = "/" + "/".join(
            str(part) for part in error.absolute_path
        )
        message = (
            "Structured generation failed: the JSON output does not conform "
            f"to the requested schema at {location or '<root>'}: "
            f"{error.message}. Verified prefix length: {len(text)} "
            "(the full document parsed, but failed schema validation)."
        )
        raise StructuredGenerationError(
            message,
            content=text,
            verified_prefix_length=len(text),
            failure_position=None,
            failure_location=location,
            reason="schema_violation",
            original_exception=error,
        )


def build_validator(output_type: Any) -> Optional[StructuredOutputValidator]:
    """Build a validator for the output type, or ``None`` if not applicable."""
    if not needs_structured_validation(output_type):
        return None
    return StructuredOutputValidator(output_type)


def validate_structured_content(output_type: Any, content: str) -> str:
    """Validate complete model output against *output_type*.

    Returns the content unchanged when valid and raises
    :class:`StructuredGenerationError` (also a ``ValueError`` and
    ``GenerationError``) otherwise. Output types that do not denote structured
    JSON are returned untouched.
    """
    validator = build_validator(output_type)
    if validator is None:
        return content
    return validator.validate_complete(content)
