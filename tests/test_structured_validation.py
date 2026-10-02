"""Tests for structured generation failure position reporting."""

from typing import Any, Iterator, List, Optional

import pytest
from pydantic import BaseModel

from outlines.exceptions import (
    GenerationError,
    OutlinesError,
    StructuredGenerationError,
)
from outlines.generator import (
    AsyncBlackBoxGenerator,
    BlackBoxGenerator,
)
from outlines.models.base import AsyncModel, Model, ModelTypeAdapter
from outlines.structured import (
    needs_structured_validation,
    validate_structured_content,
)


class _FakeTypeAdapter(ModelTypeAdapter):
    def format_input(self, model_input: Any) -> Any:
        return model_input

    def format_output_type(self, output_type: Optional[Any] = None) -> Any:
        return output_type


class FakeStreamingModel(Model):
    """Model returning scripted chunks from generate_stream."""

    type_adapter = _FakeTypeAdapter()

    def __init__(self, chunks: List[str]):
        self._chunks = list(chunks)

    def generate(self, model_input, output_type=None, **inference_kwargs):
        return "".join(self._chunks)

    def generate_batch(self, model_input, output_type=None, **inference_kwargs):
        raise NotImplementedError

    def generate_stream(
        self, model_input, output_type=None, **inference_kwargs
    ) -> Iterator[str]:
        yield from self._chunks


class _Person(BaseModel):
    name: str
    age: int


def test_needs_structured_validation():
    assert needs_structured_validation(dict)
    assert needs_structured_validation(_Person)
    from outlines.types import JsonSchema
    assert needs_structured_validation(JsonSchema(_Person.model_json_schema()))
    assert not needs_structured_validation(str)
    assert not needs_structured_validation(None)
    assert not needs_structured_validation(int)


def test_validate_complete_valid():
    text = '{"name": "Ada", "age": 27}'
    assert validate_structured_content(dict, text) == text


def test_validate_complete_plain_str_passes_through():
    assert validate_structured_content(str, "not json at all") == "not json at all"


def test_validate_complete_invalid_reports_position():
    valid_prefix = '{"name": "Ada", "age": 27, "city": "London"'
    text = valid_prefix + "!"
    with pytest.raises(StructuredGenerationError) as exc_info:
        validate_structured_content(dict, text)
    err = exc_info.value
    assert err.verified_prefix_length == len(valid_prefix)
    assert err.failure_position == len(valid_prefix)
    assert err.reason == "invalid_json_syntax"
    assert err.content == text
    assert isinstance(err, ValueError)
    assert isinstance(err, GenerationError)
    assert isinstance(err, OutlinesError)


def test_validate_complete_truncated_reports_null_position():
    text = '{"name": "Ada", "age": 27, "city": "Londo'
    with pytest.raises(StructuredGenerationError) as exc_info:
        validate_structured_content(dict, text)
    err = exc_info.value
    assert err.failure_position is None
    assert err.verified_prefix_length == len(text)
    assert err.reason == "incomplete"


def test_validate_schema_violation():
    text = '{"name": "Ada", "age": "not-an-int"}'
    with pytest.raises(StructuredGenerationError) as exc_info:
        validate_structured_content(_Person, text)
    err = exc_info.value
    assert err.reason == "schema_violation"
    assert err.failure_location == "/age"
    assert err.verified_prefix_length == len(text)


def test_schema_valid_output():
    text = '{"name": "Ada", "age": 27}'
    assert validate_structured_content(_Person, text) == text


BS = chr(92)


@pytest.mark.parametrize(
    "text,state",
    [
        ("", "incomplete"),
        ("  ", "incomplete"),
        ("{", "incomplete"),
        ('{"a"', "incomplete"),
        ('{"a":', "incomplete"),
        ('{"a":1', "incomplete"),
        ('{"a":"x', "incomplete"),
        ('{"a":"x' + BS, "incomplete"),
        ('{"a":"x' + BS + 'u00', "incomplete"),
        ("[1,", "incomplete"),
        ("tru", "incomplete"),
        ('{"a":1}', "complete"),
        ('{"a":1}  ', "complete"),
        ("true", "complete"),
        ('{"a":1}x', "invalid"),
        ('{"a":1x', "invalid"),
        ("nullx", "invalid"),
        ("12.", "invalid"),
        ('{"a":1,}', "invalid"),
        ('{"a":01}', "invalid"),
    ],
)
def test_classify_prefix_states(text, state):
    from outlines.structured import _classify_json_prefix

    assert _classify_json_prefix(text).state == state


VALID_PREFIX = '{"name": "Ada", "age": 27, "city": "London"'
ILLEGAL = "!"


def _collect(stream):
    chunks = []
    with pytest.raises(StructuredGenerationError) as exc_info:
        for chunk in stream:
            chunks.append(chunk)
    return chunks, exc_info.value


def test_stream_valid_prefix_then_illegal_char_reports_position():
    model = FakeStreamingModel([VALID_PREFIX[:20], VALID_PREFIX[20:], ILLEGAL])
    generator = BlackBoxGenerator(model, dict)

    chunks, err = _collect(generator.stream("prompt"))
    assert "".join(chunks) == VALID_PREFIX
    assert not "".join(chunks).endswith(ILLEGAL)
    assert err.verified_prefix_length == len(VALID_PREFIX)
    assert err.failure_position == len(VALID_PREFIX)
    assert err.content == VALID_PREFIX + ILLEGAL


def test_stream_truncation_reports_null_position():
    model = FakeStreamingModel([VALID_PREFIX])
    generator = BlackBoxGenerator(model, dict)

    chunks, err = _collect(generator.stream("prompt"))
    assert "".join(chunks) == VALID_PREFIX
    assert err.failure_position is None
    assert err.verified_prefix_length == len(VALID_PREFIX)


def test_stream_valid_output_passes_through():
    doc = VALID_PREFIX + "}"
    model = FakeStreamingModel([doc[:10], doc[10:25], doc[25:]])
    generator = BlackBoxGenerator(model, dict)
    assert "".join(generator.stream("prompt")) == doc


def test_stream_non_json_type_unvalidated():
    model = FakeStreamingModel(["not json", " at all"])
    generator = BlackBoxGenerator(model, str)
    assert "".join(generator.stream("prompt")) == "not json at all"


def test_call_invalid_output_raises():
    model = FakeStreamingModel([VALID_PREFIX + ILLEGAL])
    generator = BlackBoxGenerator(model, dict)
    with pytest.raises(StructuredGenerationError) as exc_info:
        generator("prompt")
    assert exc_info.value.failure_position == len(VALID_PREFIX)


def test_call_valid_output_unchanged():
    doc = VALID_PREFIX + "}"
    model = FakeStreamingModel([doc])
    generator = BlackBoxGenerator(model, dict)
    assert generator("prompt") == doc


def test_batch_validates_each_element():
    good = VALID_PREFIX + "}"
    bad = VALID_PREFIX + ILLEGAL

    class BatchModel(FakeStreamingModel):
        def generate_batch(self, model_input, output_type=None, **kwargs):
            return [good, bad]

    generator = BlackBoxGenerator(BatchModel([good]), dict)
    with pytest.raises(StructuredGenerationError) as exc_info:
        generator.batch(["p1", "p2"])
    assert exc_info.value.failure_position == len(VALID_PREFIX)


class FakeAsyncStreamingModel(AsyncModel):
    """Async model returning scripted chunks from generate_stream."""

    type_adapter = _FakeTypeAdapter()

    def __init__(self, chunks: List[str]):
        self._chunks = list(chunks)

    async def generate(self, model_input, output_type=None, **inference_kwargs):
        return "".join(self._chunks)

    async def generate_batch(self, model_input, output_type=None, **kwargs):
        raise NotImplementedError

    async def generate_stream(self, model_input, output_type=None, **kwargs):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_async_stream_valid_prefix_then_illegal_char():
    model = FakeAsyncStreamingModel(
        [VALID_PREFIX[:20], VALID_PREFIX[20:], ILLEGAL]
    )
    generator = AsyncBlackBoxGenerator(model, dict)
    chunks: List[str] = []
    with pytest.raises(StructuredGenerationError) as exc_info:
        async for chunk in generator.stream("prompt"):
            chunks.append(chunk)
    assert "".join(chunks) == VALID_PREFIX
    err = exc_info.value
    assert err.failure_position == len(VALID_PREFIX)
    assert err.verified_prefix_length == len(VALID_PREFIX)
    assert isinstance(err, ValueError)


@pytest.mark.asyncio
async def test_async_stream_valid_output():
    doc = VALID_PREFIX + "}"
    model = FakeAsyncStreamingModel([doc[:15], doc[15:]])
    generator = AsyncBlackBoxGenerator(model, dict)
    chunks: List[str] = []
    async for chunk in generator.stream("prompt"):
        chunks.append(chunk)
    assert "".join(chunks) == doc


@pytest.mark.asyncio
async def test_async_call_truncated_raises():
    model = FakeAsyncStreamingModel([VALID_PREFIX])
    generator = AsyncBlackBoxGenerator(model, dict)
    with pytest.raises(StructuredGenerationError) as exc_info:
        await generator("prompt")
    assert exc_info.value.failure_position is None
