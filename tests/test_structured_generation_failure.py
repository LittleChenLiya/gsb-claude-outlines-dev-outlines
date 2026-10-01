"""Tests for structured generation failure position reporting.

These tests use a fake model stream to verify that when structured (JSON)
generation fails, the generator reports the verified prefix length and the
failure position, and does not treat the unverified suffix as a successful
result.
"""

import pytest
from typing import Any, Iterator, List, Optional

from outlines.generator import BlackBoxGenerator, AsyncBlackBoxGenerator
from outlines.models.base import Model, ModelTypeAdapter
from outlines.exceptions import StructuredGenerationError, GenerationError


class FakeTypeAdapter(ModelTypeAdapter):
    """Minimal type adapter for the fake model."""

    def format_input(self, model_input: Any) -> Any:
        return model_input

    def format_output_type(self, output_type: Optional[Any] = None) -> Any:
        return output_type


class FakeModel(Model):
    """Fake model that returns a fixed stream of chunks."""

    type_adapter = FakeTypeAdapter()
    tensor_library_name = "numpy"

    def __init__(self, chunks: List[str]):
        self.chunks = chunks

    def generate(
        self,
        model_input: Any,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        return "".join(self.chunks)

    def generate_batch(
        self,
        model_input: List[Any],
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        return [self.generate(model_input, output_type, **kwargs)]

    def generate_stream(
        self,
        model_input: Any,
        output_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in self.chunks:
            yield chunk



class AsyncFakeModel(Model):
    type_adapter = FakeTypeAdapter()
    tensor_library_name = "numpy"

    def __init__(self, chunks):
        self.chunks = chunks

    async def generate(self, model_input, output_type=None, **kwargs):
        return "".join(self.chunks)

    async def generate_batch(self, model_input, output_type=None, **kwargs):
        return [await self.generate(model_input, output_type, **kwargs)]

    async def generate_stream(self, model_input, output_type=None, **kwargs):
        for chunk in self.chunks:
            yield chunk

class TestStructuredGenerationFailurePosition:
    """Test that structured generation failures report position info."""

    def test_invalid_character_in_stream(self):
        fake_model = FakeModel(['{"a": 1, "b": ', "}", ""])
        generator = BlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            list(generator.stream("prompt"))

        error = exc_info.value
        assert error.reason == "invalid"
        assert error.failure_position == 14
        assert error.verified_prefix_length == 14
        assert error.verified_prefix == '{"a": 1, "b": '
        assert "}" not in error.verified_prefix

    def test_incomplete_stream(self):
        fake_model = FakeModel(['{"a": 1'])
        generator = BlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            list(generator.stream("prompt"))

        error = exc_info.value
        assert error.reason == "incomplete"
        assert error.failure_position == 7
        assert error.verified_prefix_length == 7
        assert error.verified_prefix == '{"a": 1'

    def test_valid_stream_completes(self):
        fake_model = FakeModel(['{"a": ', "1}"])
        generator = BlackBoxGenerator(fake_model, dict)

        chunks = list(generator.stream("prompt"))
        assert chunks == ['{"a": ', "1}"]

    def test_non_json_output_type_not_verified(self):
        fake_model = FakeModel(["not json at all"])
        generator = BlackBoxGenerator(fake_model, str)

        chunks = list(generator.stream("prompt"))
        assert chunks == ["not json at all"]

    def test_call_with_invalid_result(self):
        fake_model = FakeModel(['{"a": 1}x'])
        generator = BlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            generator("prompt")

        error = exc_info.value
        assert error.reason == "invalid"
        assert error.failure_position == 8
        assert error.verified_prefix_length == 8
        assert error.verified_prefix == '{"a": 1}'

    def test_call_with_valid_result(self):
        fake_model = FakeModel(['{"a": 1}'])
        generator = BlackBoxGenerator(fake_model, dict)

        result = generator("prompt")
        assert result == '{"a": 1}'

    def test_exception_hierarchy_compatible(self):
        assert issubclass(StructuredGenerationError, GenerationError)
        assert issubclass(StructuredGenerationError, Exception)

    def test_error_message_contains_position_info(self):
        fake_model = FakeModel(['{"a": 1, "b": ', "}", ""])
        generator = BlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            list(generator.stream("prompt"))

        message = str(exc_info.value)
        assert "14" in message
        assert "invalid" in message


class TestAsyncStructuredGenerationFailurePosition:
    """Test async structured generation failure position reporting."""

    @pytest.mark.asyncio
    async def test_invalid_character_in_async_stream(self):
        fake_model = AsyncFakeModel(['{"a": 1, "b": ', "}", ""])
        generator = AsyncBlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            async for _ in generator.stream("prompt"):
                pass

        error = exc_info.value
        assert error.reason == "invalid"
        assert error.failure_position == 14
        assert error.verified_prefix_length == 14
        assert error.verified_prefix == '{"a": 1, "b": '

    @pytest.mark.asyncio
    async def test_incomplete_async_stream(self):
        fake_model = AsyncFakeModel(['{"a": 1'])
        generator = AsyncBlackBoxGenerator(fake_model, dict)

        with pytest.raises(StructuredGenerationError) as exc_info:
            async for _ in generator.stream("prompt"):
                pass

        error = exc_info.value
        assert error.reason == "incomplete"
        assert error.failure_position == 7
        assert error.verified_prefix_length == 7

    @pytest.mark.asyncio
    async def test_valid_async_stream_completes(self):
        fake_model = AsyncFakeModel(['{"a": ', "1}"])
        generator = AsyncBlackBoxGenerator(fake_model, dict)

        chunks = []
        async for chunk in generator.stream("prompt"):
            chunks.append(chunk)
        assert chunks == ['{"a": ', "1}"]
