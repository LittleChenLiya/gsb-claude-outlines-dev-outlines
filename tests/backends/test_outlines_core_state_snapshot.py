"""Tests for the serializable state snapshots of the Outlines Core backend.

These tests use a small synthetic vocabulary (the ten digits plus an EOS
token) and a fixed grammar so they run locally without downloading any
model. For the fixed grammar, an input is fed in two segments and the
accept states of continuous parsing and snapshot-restored parsing are
compared.
"""

import json

import numpy as np
import pytest
from outlines_core import Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor

REGEX = r"[0-9]{3}"
OTHER_REGEX = r"[0-9]{2}"
EOS_TOKEN_ID = 10
VOCAB_SIZE = 11  # 10 digit tokens + EOS
BATCH_SIZE = 2

# Token ids fed to the processors, one per row of the batch. The first
# token of each row stands for the prompt (it is never advanced by the
# processor); the three following tokens are the generated digits.
ROW_TOKENS = [[0, 1, 2, 3], [5, 6, 7, 8]]
# The snapshot is taken after this many generation steps, splitting the
# input into two segments.
SPLIT_AT = 2


def make_processor(grammar=REGEX):
    vocabulary = Vocabulary(EOS_TOKEN_ID, {str(d): [d] for d in range(10)})
    index = Index(grammar, vocabulary)
    return OutlinesCoreLogitsProcessor(index, "numpy", grammar)


def make_logits():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((BATCH_SIZE, VOCAB_SIZE)) for _ in ROW_TOKENS[0]]


def feed_step(processor, logits, step):
    """Feed generation step `step` (1-based) of the input."""
    input_ids = np.array([row[:step] for row in ROW_TOKENS])
    return processor.process_logits(input_ids, logits[step - 1])


def run_continuous(processor, logits):
    """Feed the full input in one continuous run, return final logits."""
    for step in range(1, len(ROW_TOKENS[0])):
        feed_step(processor, logits, step)
    return feed_step(processor, logits, len(ROW_TOKENS[0]))


def run_resumed(processor, logits):
    """Feed the input in two segments, snapshotting in between.

    Returns the resumed processor and the final biased logits.
    """
    for step in range(1, SPLIT_AT + 1):
        feed_step(processor, logits, step)
    snapshot = processor.get_state_snapshot()
    # The snapshot must be JSON-serializable so it can be stored or sent
    # over the wire for the same request to be resumed later.
    snapshot = json.loads(json.dumps(snapshot))

    resumed = make_processor()
    resumed.load_state_snapshot(snapshot)
    # The resumed processor only receives the second segment; the last
    # token of each row is the newly generated one.
    final_logits = None
    for step in range(SPLIT_AT + 1, len(ROW_TOKENS[0]) + 1):
        final_logits = feed_step(resumed, logits, step)
    return resumed, final_logits


def assert_same_accept_state(resumed, continuous):
    for i in range(BATCH_SIZE):
        # The accept states must be identical.
        assert resumed._guides[i].get_state() == continuous._guides[i].get_state()
        assert resumed._guides[i].is_finished() == continuous._guides[i].is_finished()
        # The sets of allowed tokens must be identical.
        assert resumed._guides[i].get_tokens() == continuous._guides[i].get_tokens()
        # The recorded token histories must be identical.
        assert resumed._token_histories[i] == continuous._token_histories[i]


def test_snapshot_is_json_serializable():
    processor = make_processor()
    feed_step(processor, make_logits(), 1)
    snapshot = processor.get_state_snapshot()
    assert snapshot["format"] == "outlines-core-state-v1"
    assert snapshot["grammar_fingerprint"] is not None
    json.dumps(snapshot)


def test_resumed_parsing_matches_continuous_parsing():
    logits = make_logits()

    continuous = make_processor()
    continuous_logits = run_continuous(continuous, logits)

    first_segment = make_processor()
    resumed, resumed_logits = run_resumed(first_segment, logits)

    assert_same_accept_state(resumed, continuous)
    # The three generated digits were advanced, so the grammar is satisfied.
    for i in range(BATCH_SIZE):
        assert resumed._guides[i].is_finished()
    # The biased logits produced after the full input must be identical.
    np.testing.assert_array_equal(resumed_logits, continuous_logits)


def test_load_snapshot_into_already_setup_processor():
    logits = make_logits()

    continuous = make_processor()
    continuous_logits = run_continuous(continuous, logits)

    first_segment = make_processor()
    for step in range(1, SPLIT_AT + 1):
        feed_step(first_segment, logits, step)
    snapshot = first_segment.get_state_snapshot()

    # This processor is already set up (it processed a first token) when
    # the snapshot is loaded.
    resumed = make_processor()
    feed_step(resumed, logits, 1)
    resumed.load_state_snapshot(snapshot)
    resumed_logits = None
    for step in range(SPLIT_AT + 1, len(ROW_TOKENS[0]) + 1):
        resumed_logits = feed_step(resumed, logits, step)

    assert_same_accept_state(resumed, continuous)
    np.testing.assert_array_equal(resumed_logits, continuous_logits)


def test_snapshot_of_fresh_processor_restores_as_fresh():
    logits = make_logits()
    snapshot = make_processor().get_state_snapshot()
    assert snapshot["sequences"] == []

    continuous = make_processor()
    continuous_logits = run_continuous(continuous, logits)

    resumed = make_processor()
    resumed.load_state_snapshot(snapshot)
    resumed_logits = run_continuous(resumed, logits)

    assert_same_accept_state(resumed, continuous)
    np.testing.assert_array_equal(resumed_logits, continuous_logits)


def test_snapshot_from_different_grammar_is_rejected():
    processor = make_processor()
    feed_step(processor, make_logits(), 1)
    snapshot = processor.get_state_snapshot()

    other = make_processor(grammar=OTHER_REGEX)
    with pytest.raises(ValueError, match="different grammar"):
        other.load_state_snapshot(snapshot)


def test_snapshot_cannot_be_verified_without_grammar():
    processor = make_processor()
    feed_step(processor, make_logits(), 1)
    snapshot = processor.get_state_snapshot()

    vocabulary = Vocabulary(EOS_TOKEN_ID, {str(d): [d] for d in range(10)})
    no_grammar = OutlinesCoreLogitsProcessor(Index(REGEX, vocabulary), "numpy")
    with pytest.raises(ValueError, match="different grammar"):
        no_grammar.load_state_snapshot(snapshot)


def test_invalid_snapshots_are_rejected():
    processor = make_processor()
    with pytest.raises(ValueError, match="Invalid state snapshot"):
        processor.load_state_snapshot({"format": "something-else"})
    with pytest.raises(ValueError, match="Invalid state snapshot"):
        processor.load_state_snapshot("not a dict")

    valid = processor.get_state_snapshot()
    invalid = {**valid, "sequences": [["not-a-token-id"]]}
    with pytest.raises(ValueError, match="Invalid state snapshot"):
        processor.load_state_snapshot(invalid)


def test_snapshot_with_mismatched_batch_size_is_rejected():
    logits = make_logits()
    processor = make_processor()
    feed_step(processor, logits, 1)
    feed_step(processor, logits, 2)
    snapshot = processor.get_state_snapshot()

    resumed = make_processor()
    resumed.load_state_snapshot(snapshot)
    with pytest.raises(ValueError, match="batch size"):
        resumed.process_logits(np.array([[2, 3]]), logits[2][:1])
