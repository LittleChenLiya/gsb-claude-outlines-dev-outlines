"""Constraint state snapshot / restore for incremental parsing.

This example demonstrates the serializable state snapshot capability of the
Outlines Core constraint backend. For a fixed grammar it feeds the same token
sequence in two ways and compares the acceptance state:

- continuously: one processor consumes the whole sequence;
- restored: a processor consumes the first segment, snapshots its state, a
  second processor restores the snapshot and consumes the second segment.

It also shows that a snapshot is JSON-serializable and that restoring a
snapshot built for a different grammar is rejected.

Run it from the root of the repository with the local source on the path:

    PYTHONPATH=src python examples/constraint_state_snapshot.py

The script prints the acceptance state after every step and exits with a
non-zero status if the restored states do not match the continuous ones.
"""

import json
import sys

import numpy as np
from outlines_core import Index, Vocabulary

from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor


# Fixed grammar: three digits.
REGEX = r"[0-9]{3}"
# Fixed vocabulary: token ids 0..9 are the digits, 10 is the EOS token.
EOS_TOKEN_ID = 10
VOCAB_DICT = {str(i): [i] for i in range(10)}
VOCAB_SIZE = len(VOCAB_DICT) + 1

# A fixed vocabulary fingerprint shared by all processors built for the same
# grammar so their grammar hashes match.
VOCABULARY_FINGERPRINT = "example-vocabulary"


def make_processor(regex: str = REGEX) -> OutlinesCoreLogitsProcessor:
    vocab = Vocabulary(EOS_TOKEN_ID, dict(VOCAB_DICT))
    index = Index(regex, vocab)
    return OutlinesCoreLogitsProcessor(
        index,
        "numpy",
        grammar=regex,
        vocabulary_fingerprint=VOCABULARY_FINGERPRINT,
    )


def drive(processor, initial_input_ids, token_sequence):
    """Drive the processor through token_sequence and return the acceptance
    state after each call."""
    states = []
    input_ids = np.array(initial_input_ids, dtype=np.int64)
    logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    processor(input_ids, logits)
    guide = processor._guides[0]
    states.append((guide.is_finished(), sorted(guide.get_tokens())))
    for token_id in token_sequence:
        input_ids = np.concatenate(
            [input_ids, np.array([[token_id]], dtype=np.int64)], axis=1
        )
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        processor(input_ids, logits)
        guide = processor._guides[0]
        states.append((guide.is_finished(), sorted(guide.get_tokens())))
    return states


def main():
    token_sequence = [1, 2, 3]  # "123"
    segment_1 = token_sequence[:2]
    segment_2 = token_sequence[2:]

    # Continuous parsing of the whole sequence.
    continuous = make_processor()
    continuous_states = drive(continuous, [[EOS_TOKEN_ID]], token_sequence)

    # Parsing of the first segment, then snapshot and restore.
    first = make_processor()
    first_states = drive(first, [[EOS_TOKEN_ID]], segment_1)
    snapshot = first.snapshot()

    restored = make_processor()
    restored.restore(snapshot)
    restored_states = drive(restored, [[EOS_TOKEN_ID] + segment_1], segment_2)

    print("Grammar:", REGEX)
    print("Token sequence:", token_sequence)
    print()
    print("Continuous acceptance states (is_finished, allowed_tokens):")
    for i, state in enumerate(continuous_states):
        print(f"  step {i}: {state}")
    print()
    print("Snapshot (JSON-serializable):")
    print(" ", json.dumps(snapshot))
    print()
    print("Restored acceptance states (is_finished, allowed_tokens):")
    for i, state in enumerate(restored_states):
        print(f"  step {i}: {state}")
    print()

    # Compare.
    ok = True
    if restored_states[0] != first_states[-1]:
        print("MISMATCH: restored state at split point != first segment state")
        ok = False
    if restored_states[0] != continuous_states[len(segment_1)]:
        print("MISMATCH: restored state at split point != continuous state")
        ok = False
    if restored_states[-1] != continuous_states[-1]:
        print("MISMATCH: final restored state != final continuous state")
        ok = False
    for i, state in enumerate(restored_states):
        if state != continuous_states[len(segment_1) + i]:
            print(f"MISMATCH at step {i}: {state} != {continuous_states[len(segment_1) + i]}")
            ok = False

    if ok:
        print("SUCCESS: restored acceptance states match continuous parsing.")
    else:
        print("FAILURE: restored states do not match continuous parsing.")
        sys.exit(1)

    # Boundary: a snapshot built for a different grammar must be rejected.
    other = make_processor(regex=r"[0-9]{2}")
    try:
        other.restore(snapshot)
    except ValueError as exc:
        print()
        print("Boundary check passed: different grammar rejected ->")
        print(" ", exc)
    else:
        print("FAILURE: different grammar was not rejected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
