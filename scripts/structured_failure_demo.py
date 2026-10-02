"""Acceptance demo: structured generation failure position reporting.

A fake black-box model streams three stages:
1. a valid JSON prefix,
2. an illegal character that makes the output irrecoverable,
3. end of stream.

The demo checks that:
* the failure raises ``StructuredGenerationError``;
* the exception reports the verified prefix length and the failure position;
* only the verified prefix was ever delivered (unverified suffix withheld);
* the old exception types stay compatible (it is a ``ValueError`` and a
  ``GenerationError``).
"""

from typing import Any, List, Optional

from outlines.exceptions import GenerationError, StructuredGenerationError
from outlines.generator import BlackBoxGenerator
from outlines.models.base import Model, ModelTypeAdapter


class _FakeTypeAdapter(ModelTypeAdapter):
    def format_input(self, model_input: Any) -> Any:
        return model_input

    def format_output_type(self, output_type: Optional[Any] = None) -> Any:
        return output_type


class FakeStreamingModel(Model):
    """Black-box-style model whose stream returns scripted chunks."""

    type_adapter = _FakeTypeAdapter()

    def __init__(self, chunks: List[str]):
        self._chunks = list(chunks)

    def generate(self, model_input, output_type=None, **inference_kwargs):
        return "".join(self._chunks)

    def generate_batch(self, model_input, output_type=None, **inference_kwargs):
        raise NotImplementedError

    def generate_stream(self, model_input, output_type=None, **inference_kwargs):
        for chunk in self._chunks:
            yield chunk


def main() -> None:
    # The model emits a valid JSON object prefix and then an illegal token
    # (after a closed string, ',' or '}' is required, so '!' is invalid).
    valid_prefix = '{"name": "Ada", "age": 27, "city": "London"'
    illegal_char = "!"
    chunks = [valid_prefix[:20], valid_prefix[20:], illegal_char]

    model = FakeStreamingModel(chunks)
    generator = BlackBoxGenerator(model, dict)

    received: List[str] = []
    print("=== Streaming structured generation (dict/JSON mode) ===")
    try:
        for chunk in generator.stream("emit a person object"):
            received.append(chunk)
            print(f"  yielded chunk: {chunk!r}")
        full = "".join(received)
        print("  stream completed with:", repr(full))
    except StructuredGenerationError as exc:
        delivered = "".join(received)
        print("  !! StructuredGenerationError raised")
        print("     message               :", exc)
        print("     verified_prefix_length:", exc.verified_prefix_length)
        print("     failure_position      :", exc.failure_position)
        print("     reason                :", exc.reason)
        print("     delivered to consumer :", repr(delivered))
        print("     withheld suffix       :",
              repr(exc.content[exc.verified_prefix_length:]))

        # --- assertions that make the behavior difference explicit ---------
        assert exc.failure_position == len(valid_prefix), (
            exc.failure_position, len(valid_prefix)
        )
        assert exc.verified_prefix_length == len(valid_prefix), (
            exc.verified_prefix_length, len(valid_prefix)
        )
        assert delivered == valid_prefix, (delivered, valid_prefix)
        assert not delivered.endswith(illegal_char)
        # Old exception types remain compatible.
        assert isinstance(exc, ValueError)
        assert isinstance(exc, GenerationError)
        print("  assertions OK: position and verified prefix length correct,")
        print("  unverified suffix was never yielded, exception types compat.")

    # --- non-streaming call reports the same location ---------------------
    print()
    print("=== Non-streaming structured generation ===")
    bad_full = valid_prefix + illegal_char
    model_full = FakeStreamingModel([bad_full])
    generator_full = BlackBoxGenerator(model_full, dict)
    try:
        result = generator_full("emit a person object")
        print("  unexpected success:", repr(result))
    except StructuredGenerationError as exc:
        print("  StructuredGenerationError raised")
        print("     verified_prefix_length:", exc.verified_prefix_length)
        print("     failure_position      :", exc.failure_position)
        assert isinstance(exc, ValueError)
        assert exc.failure_position == len(valid_prefix)
        print("  assertions OK.")

    # --- truncated output: failure_position is None (ended early) ---------
    print()
    print("=== Truncated output (stream ends mid-document) ===")
    model_trunc = FakeStreamingModel([valid_prefix])
    generator_trunc = BlackBoxGenerator(model_trunc, dict)
    received_trunc: List[str] = []
    try:
        for chunk in generator_trunc.stream("emit a person object"):
            received_trunc.append(chunk)
        print("  unexpected success:", repr("".join(received_trunc)))
    except StructuredGenerationError as exc:
        print("  StructuredGenerationError raised")
        print("     verified_prefix_length:", exc.verified_prefix_length)
        print("     failure_position      :", exc.failure_position)
        assert exc.failure_position is None
        assert exc.verified_prefix_length == len(valid_prefix)
        assert "".join(received_trunc) == valid_prefix
        print("  assertions OK: early end reported with null position,")
        print("  and the verified prefix was still delivered.")

    # --- valid output still succeeds --------------------------------------
    print()
    print("=== Valid output ===")
    valid_doc = '{"name": "Ada", "age": 27, "city": "London"}'
    model_ok = FakeStreamingModel([valid_doc[:15], valid_doc[15:30], valid_doc[30:]])
    result = BlackBoxGenerator(model_ok, dict)("emit a person object")
    assert result == valid_doc
    print("  full result:", result)

    chunks_ok: List[str] = []
    for chunk in BlackBoxGenerator(FakeStreamingModel([valid_doc[:15], valid_doc[15:30], valid_doc[30:]]), dict).stream("x"):
        chunks_ok.append(chunk)
    assert "".join(chunks_ok) == valid_doc
    print("  streamed  :", "".join(chunks_ok))
    print("  assertions OK: valid structured output is unaffected.")


if __name__ == "__main__":
    main()
