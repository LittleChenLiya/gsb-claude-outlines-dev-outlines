"""Tests for the constraint state snapshot / restore capability of the
Outlines Core backend.

These tests construct the logits processors directly from a fixed grammar and
a fixed vocabulary (no model download is required) and drive them with numpy
tensors. The same token sequence is fed in two ways:

- continuously: a single processor consumes the whole sequence;
- restored: a processor consumes the first segment, snapshots its state, a
  second processor restores the snapshot and consumes the second segment.

The acceptance state (``is_finished`` and the set of allowed tokens) and the
biased logits mask are compared at every step to verify that restoring a
snapshot resumes incremental parsing exactly where it left off.
"""

import json

import numpy as np
import pytest
from outlines_core import Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor


# A fixed grammar: three digits.
REGEX = r"[0-9]{3}"
# A fixed vocabulary: token ids 0..9 are the digits, 10 is the EOS token.
EOS_TOKEN_ID = 10
VOCAB_DICT = {str(i): [i] for i in range(10)}
VOCAB_SIZE = len(VOCAB_DICT) + 1  # digits + eos

# A fixed fingerprint for the vocabulary, shared by all processors built for
# the same grammar so their grammar hashes match.
VOCABULARY_FINGERPRINT = "test-vocabulary"


def make_processor(regex: str = REGEX) -> OutlinesCoreLogitsProcessor:
    """Build a logits processor for the given grammar."""
    vocab = Vocabulary(EOS_TOKEN_ID, dict(VOCAB_DICT))
    index = Index(regex, vocab)
    return OutlinesCoreLogitsProcessor(
        index,
        "numpy",
        grammar=regex,
        vocabulary_fingerprint=VOCABULARY_FINGERPRINT,
    )


def drive(
    processor: OutlinesCoreLogitsProcessor,
    initial_input_ids,
    token_sequence,
):
    """Drive the processor through ``token_sequence``.

    The first call uses ``initial_input_ids`` and does not advance the guide
    (it only biases the logits). Each subsequent call appends one token and
    advances the guide by that token.

    Returns a list of ``(is_finished, allowed_tokens, mask)`` tuples, one per
    call, where ``mask`` is the boolean array of tokens allowed by the biased
    logits.
    """
    states = []
    input_ids = np.array(initial_input_ids, dtype=np.int64)
    logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    processor(input_ids, logits)
    guide = processor._guides[0]
    states.append(
        (
            guide.is_finished(),
            sorted(guide.get_tokens()),
            np.isfinite(logits[0]).copy(),
        )
    )
    for token_id in token_sequence:
        input_ids = np.concatenate(
            [input_ids, np.array([[token_id]], dtype=np.int64)], axis=1
        )
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        processor(input_ids, logits)
        guide = processor._guides[0]
        states.append(
            (
                guide.is_finished(),
                sorted(guide.get_tokens()),
                np.isfinite(logits[0]).copy(),
            )
        )
    return states


def states_equal(state_a, state_b) -> bool:
    """Compare two ``(is_finished, allowed_tokens, mask)`` states."""
    return (
        state_a[0] == state_b[0]
        and state_a[1] == state_b[1]
        and np.array_equal(state_a[2], state_b[2])
    )


def test_snapshot_restore_matches_continuous_parsing():
    """Restoring a snapshot after the first segment yields the same acceptance
    state as continuous parsing of the whole sequence."""
    token_sequence = [1, 2, 3]  # "123"
    segment_1 = token_sequence[:2]
    segment_2 = token_sequence[2:]

    # Continuous parsing of the whole sequence.
    processor_continuous = make_processor()
    states_continuous = drive(processor_continuous, [[EOS_TOKEN_ID]], token_sequence)

    # Parsing of the first segment, then snapshot and restore.
    processor_segment_1 = make_processor()
    states_segment_1 = drive(processor_segment_1, [[EOS_TOKEN_ID]], segment_1)
    snapshot = processor_segment_1.snapshot()

    processor_restored = make_processor()
    processor_restored.restore(snapshot)
    states_restored = drive(
        processor_restored, [[EOS_TOKEN_ID] + segment_1], segment_2
    )

    # The restored state at the split point must match the continuous state
    # after consuming the first segment.
    assert states_equal(states_restored[0], states_segment_1[-1])
    assert states_equal(states_restored[0], states_continuous[len(segment_1)])

    # The final restored state must match the final continuous state.
    assert states_equal(states_restored[-1], states_continuous[-1])

    # Every restored state must match its continuous counterpart.
    for i, state in enumerate(states_restored):
        assert states_equal(state, states_continuous[len(segment_1) + i])

    # The final state accepts the EOS token and no digit is allowed.
    final = states_continuous[-1]
    assert final[0] is True
    assert final[1] == [EOS_TOKEN_ID]
    assert np.array_equal(final[2], np.arange(VOCAB_SIZE) == EOS_TOKEN_ID)


def test_snapshot_is_json_serializable():
    """The snapshot can be serialized to JSON and restored from the parsed
    object."""
    processor = make_processor()
    drive(processor, [[EOS_TOKEN_ID]], [1, 2])
    snapshot = processor.snapshot()

    serialized = json.dumps(snapshot)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["version"] == 1
    assert deserialized["grammar"] == REGEX
    assert deserialized["tokens"] == [[1, 2]]

    restored = make_processor()
    restored.restore(deserialized)
    states = drive(restored, [[EOS_TOKEN_ID, 1, 2]], [3])
    assert states[-1][0] is True


def test_snapshot_tokens_match_consumed_tokens():
    """The snapshot records the exact token prefix consumed by the guide."""
    processor = make_processor()
    drive(processor, [[EOS_TOKEN_ID]], [1, 2, 3])
    snapshot = processor.snapshot()
    assert snapshot["tokens"] == [[1, 2, 3]]


def test_restore_rejects_different_grammar():
    """A snapshot built for a different grammar must be rejected."""
    processor = make_processor(regex=r"[0-9]{3}")
    drive(processor, [[EOS_TOKEN_ID]], [1, 2])
    snapshot = processor.snapshot()

    # A different grammar that is still compatible with the digit vocabulary.
    other = make_processor(regex=r"[0-9]{2}")
    with pytest.raises(ValueError, match="grammar"):
        other.restore(snapshot)


def test_restore_rejects_unsupported_version():
    """A snapshot with an unsupported version must be rejected."""
    processor = make_processor()
    snapshot = processor.snapshot()
    snapshot["version"] = 0
    with pytest.raises(ValueError, match="Unsupported snapshot version"):
        processor.restore(snapshot)


def test_restore_without_grammar_identity():
    """A processor built without grammar information can still snapshot and
    restore, but it rejects snapshots that carry a grammar identity."""
    vocab = Vocabulary(EOS_TOKEN_ID, dict(VOCAB_DICT))
    index = Index(REGEX, vocab)
    processor = OutlinesCoreLogitsProcessor(index, "numpy")
    drive(processor, [[EOS_TOKEN_ID]], [1, 2])
    snapshot = processor.snapshot()
    assert snapshot["grammar"] is None
    assert snapshot["grammar_hash"] is None

    # Restoring on another processor without grammar information works.
    other = OutlinesCoreLogitsProcessor(Index(REGEX, vocab), "numpy")
    other.restore(snapshot)
    states = drive(other, [[EOS_TOKEN_ID, 1, 2]], [3])
    assert states[-1][0] is True

    # Restoring a snapshot that carries a grammar identity on a processor
    # without one is rejected.
    with_grammar = make_processor()
    drive(with_grammar, [[EOS_TOKEN_ID]], [1, 2])
    with pytest.raises(ValueError, match="grammar identity"):
        processor.restore(with_grammar.snapshot())


def test_reset_preserves_pending_restore():
    """Calling ``reset`` after ``restore`` must not discard the pending
    restore: the next call to ``process_logits`` replays the token prefix."""
    processor = make_processor()
    drive(processor, [[EOS_TOKEN_ID]], [1, 2])
    snapshot = processor.snapshot()

    restored = make_processor()
    restored.restore(snapshot)
    restored.reset()  # this is what the generator does before generating

    states = drive(restored, [[EOS_TOKEN_ID, 1, 2]], [3])
    # The first state after the reset must be the restored state (after
    # consuming the first segment), not the initial state.
    assert states[0][0] is False
    assert states[0][1] == sorted(range(10))
    assert states[-1][0] is True


def test_restore_then_snapshot_is_idempotent():
    """Snapshotting a restored processor yields the same snapshot."""
    processor = make_processor()
    drive(processor, [[EOS_TOKEN_ID]], [1, 2])
    snapshot = processor.snapshot()

    restored = make_processor()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot


def test_snapshot_before_any_call_is_empty():
    """A snapshot taken before any call to ``process_logits`` is empty."""
    processor = make_processor()
    snapshot = processor.snapshot()
    assert snapshot["tokens"] == []

    restored = make_processor()
    restored.restore(snapshot)
    states = drive(restored, [[EOS_TOKEN_ID]], [1, 2, 3])
    assert states[-1][0] is True
