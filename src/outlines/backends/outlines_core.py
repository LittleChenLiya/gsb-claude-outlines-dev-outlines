"""Backend class for Outlines Core."""

import hashlib
from typing import Any, Callable, Dict, List, Optional

from outlines_core import Guide, Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema

from outlines.backends.base import BaseBackend
from outlines.models import SteerableModel
from outlines.models.llamacpp import LlamaCpp
from outlines.models.mlxlm import MLXLM
from outlines.models.transformers import Transformers
from outlines.processors.base_logits_processor import (
    OutlinesLogitsProcessor,
    TensorType,
)

SNAPSHOT_FORMAT_VERSION = "outlines-core-state-v1"


def _grammar_fingerprint(grammar: str) -> str:
    """Compute a stable fingerprint for a grammar (regex) string."""
    return hashlib.sha256(grammar.encode("utf-8")).hexdigest()


class OutlinesCoreLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor for Outlines Core."""

    def __init__(
        self,
        index: Index,
        tensor_library_name: str,
        grammar: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        index: Index
            The Outlines Core `Index` instance to use to create the Outlines
            Core `Guide` instances that will be used to bias the logits
        tensor_library_name: str
            The tensor library name to use for the logits processor.
        grammar: str | None
            The grammar (regex) the `index` was built from. It is used to
            fingerprint state snapshots so that a snapshot cannot be
            restored into a processor that enforces a different grammar.

        """
        self.index = index
        self.tensor_library_name = tensor_library_name
        self.grammar_fingerprint = (
            _grammar_fingerprint(grammar) if grammar is not None else None
        )
        self.is_first_token = True
        self._token_histories: Optional[List[List[int]]] = None
        self._pending_restore: Optional[List[List[int]]] = None
        self._restored = False
        super().__init__(tensor_library_name)

    def reset(self) -> None:
        """Reset the logits processor."""
        self.is_first_token = True
        self._token_histories = None
        self._pending_restore = None
        self._restored = False

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current parsing state.

        The snapshot is a plain dictionary of JSON-serializable values. It
        can be stored or transmitted and later passed to
        `load_state_snapshot` of a processor built for the same grammar to
        resume incremental parsing of the same request.

        Returns
        -------
        Dict[str, Any]
            The state snapshot.

        """
        return {
            "format": SNAPSHOT_FORMAT_VERSION,
            "grammar_fingerprint": self.grammar_fingerprint,
            "sequences": (
                [list(history) for history in self._token_histories]
                if self._token_histories is not None
                else []
            ),
        }

    def load_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore the parsing state from a snapshot.

        The snapshot must have been created by `get_state_snapshot` on a
        processor enforcing the same grammar, otherwise a `ValueError` is
        raised. After loading, the next calls to `process_logits` behave as
        if the tokens recorded in the snapshot had already been processed.

        Parameters
        ----------
        snapshot: Dict[str, Any]
            The snapshot to restore, as returned by `get_state_snapshot`.

        """
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("format") != SNAPSHOT_FORMAT_VERSION
        ):
            raise ValueError(
                "Invalid state snapshot: expected a dictionary with "
                f"'format' equal to {SNAPSHOT_FORMAT_VERSION!r}."
            )
        if (
            self.grammar_fingerprint is None
            or snapshot.get("grammar_fingerprint") != self.grammar_fingerprint
        ):
            raise ValueError(
                "State snapshot was created for a different grammar and "
                "cannot be restored into this processor."
            )
        sequences = snapshot.get("sequences")
        if not isinstance(sequences, list) or not all(
            isinstance(seq, list) and all(isinstance(t, int) for t in seq)
            for seq in sequences
        ):
            raise ValueError(
                "Invalid state snapshot: 'sequences' must be a list of "
                "lists of token ids."
            )

        if not sequences:
            # Snapshot of a processor that has not processed anything yet:
            # restoring it is equivalent to a fresh start.
            self._pending_restore = None
            self._restored = False
            return
        self._pending_restore = [list(seq) for seq in sequences]
        self._restored = True
        if self._token_histories is not None:
            # The processor is already set up: apply the snapshot now.
            self._apply_pending_restore(len(self._token_histories))

    def _apply_pending_restore(self, batch_size: int) -> None:
        """Replay the token histories of a loaded snapshot on the guides."""
        histories = self._pending_restore
        if histories is None:
            return
        if len(histories) != batch_size:
            raise ValueError(
                f"State snapshot contains {len(histories)} sequence(s) but "
                f"the current batch size is {batch_size}."
            )
        self._token_histories = []
        for i, history in enumerate(histories):
            self._guides[i].reset()
            for token_id in history:
                self._guides[i].advance(token_id=token_id, return_tokens=False)
            self._token_histories.append(list(history))
        self._pending_restore = None

    def _setup(self, batch_size: int, vocab_size: int) -> None:
        """Set the guides, bitmasks and some functions used in the
        `process_logits` method.

        This method is called when the first token is generated instead of
        at initialization because we need to know the batch size and the device
        of the logits.

        Parameters
        ----------
        batch_size: int
            The batch size.
        vocab_size: int
            The vocabulary size.

        """
        if self.tensor_library_name == "torch":
            from outlines_core.kernels.torch import allocate_token_bitmask

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_torch

        elif self.tensor_library_name == "numpy":
            from outlines_core.kernels.numpy import allocate_token_bitmask

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_numpy

        elif self.tensor_library_name == "mlx":  # pragma: no cover
            from outlines_core.kernels.mlx import allocate_token_bitmask

            self.allocate_token_bitmask = allocate_token_bitmask
            self.bias_logits = self._bias_logits_mlx

        else:  # pragma: no cover
            raise ValueError(f"Unsupported tensor library: {self.tensor_library_name}")

        self._guides = [Guide(self.index) for _ in range(batch_size)]
        self._bitmasks = [
            self.allocate_token_bitmask(vocab_size) for _ in range(batch_size)
        ]
        self._token_histories = [[] for _ in range(batch_size)]
        self._apply_pending_restore(batch_size)

    def _bias_logits_mlx(  # pragma: no cover
        self, batch_size: int, logits: TensorType
    ) -> TensorType:
        """Bias the logits for MLX tensors."""
        from outlines_core.kernels.mlx import (
            apply_token_bitmask,
            fill_next_token_bitmask,
        )

        biased_logits_array = []
        for i in range(batch_size):
            fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
            biased_logits = apply_token_bitmask(
                self.tensor_adapter.unsqueeze(logits[i]), # type: ignore
                self._bitmasks[i],  # type: ignore
            )
            biased_logits_array.append(biased_logits)

        return self.tensor_adapter.concatenate(biased_logits_array)

    def _bias_logits_torch(self, batch_size: int, logits: TensorType) -> TensorType:
        """Bias the logits for Torch tensors."""
        from outlines_core.kernels.torch import (
            apply_token_bitmask_inplace,
            fill_next_token_bitmask,
        )

        for i in range(batch_size):
            fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
            self._bitmasks[i] = self.tensor_adapter.to_device(
                self._bitmasks[i], self.tensor_adapter.get_device(logits)
            )
            apply_token_bitmask_inplace(
                self.tensor_adapter.unsqueeze(logits[i]),  # type: ignore
                self._bitmasks[i],
            )
            self._bitmasks[i] = self.tensor_adapter.to_device(self._bitmasks[i], "cpu")

        return logits

    def _bias_logits_numpy(self, batch_size: int, logits: TensorType) -> TensorType:
        """Bias the logits for Numpy tensors."""
        from outlines_core.kernels.numpy import (
            apply_token_bitmask_inplace,
            fill_next_token_bitmask,
        )

        for i in range(batch_size):
            fill_next_token_bitmask(self._guides[i], self._bitmasks[i])
            apply_token_bitmask_inplace(
                self.tensor_adapter.unsqueeze(logits[i]),  # type: ignore
                self._bitmasks[i],
            )

        return logits

    def process_logits(self, input_ids: TensorType, logits: TensorType) -> TensorType:
        """Use the guides to bias the logits.

        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        batch_size = self.tensor_adapter.shape(input_ids)[0]
        vocab_size = self.tensor_adapter.shape(logits)[1]

        if self.is_first_token:
            self._setup(batch_size, vocab_size)
            self.is_first_token = False
            # A processor restored from a snapshot continues an existing
            # sequence, so the last token of `input_ids` must be advanced
            # like for any subsequent call.
            should_advance = self._restored
        else:
            should_advance = True

        if should_advance:
            for i in range(batch_size):
                last_token_id = self.tensor_adapter.to_scalar(input_ids[i][-1])  # type: ignore
                # This circumvents issue #227 in outlines_core
                # Ideally, we would be able to advance all the times as the final
                # state would accept the eos token leading to itself
                if not self._guides[i].is_finished() or self._guides[i].accepts_tokens(
                    [last_token_id]
                ):
                    self._guides[i].advance(token_id=last_token_id, return_tokens=False)
                    self._token_histories[i].append(last_token_id)  # type: ignore

        return self.bias_logits(batch_size, logits)


class OutlinesCoreBackend(BaseBackend):
    """Backend for Outlines Core."""

    def __init__(self, model: SteerableModel):
        """
        Parameters
        ----------
        model
            The Outlines model of the user.

        """
        if isinstance(model, Transformers):
            tokenizer = model.tokenizer
            vocabulary = tokenizer.get_vocab()
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
            token_to_str = tokenizer.convert_token_to_string
        elif isinstance(model, LlamaCpp):
            tokenizer = model.tokenizer  # type: ignore
            vocabulary = tokenizer.vocabulary
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
            token_to_str = tokenizer.convert_token_to_string
        elif isinstance(model, MLXLM):  # pragma: no cover
            tokenizer = model.mlx_tokenizer  # type: ignore
            vocabulary = tokenizer.get_vocab()
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token
            token_to_str = lambda token: tokenizer.convert_tokens_to_string([token])  # type: ignore
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model type: {type(model)}")

        self.eos_token_id = eos_token_id
        self.vocabulary = self.create_outlines_core_vocabulary(
            vocabulary, eos_token_id, eos_token, token_to_str
        )
        self.tensor_library_name = model.tensor_library_name

    def get_json_schema_logits_processor(
        self, json_schema: str, whitespace_pattern: str | None = None
    ):
        """Create a logits processor from a JSON schema.

        Parameters
        ----------
        json_schema: str
            The JSON schema to create a logits processor from.
        whitespace_pattern: str | None
            The pattern to use to control the whitespace allowed between JSON
            tokens. `None` uses the outlines_core default.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        regex = build_regex_from_schema(json_schema, whitespace_pattern)
        return self.get_regex_logits_processor(regex)

    def get_regex_logits_processor(self, regex: str):
        """Create a logits processor from a regex.

        Parameters
        ----------
        regex: str
            The regex to create a logits processor from.

        Returns
        -------
        LogitsProcessor
            The logits processor to use to constrain the generation.

        """
        index = Index(regex, self.vocabulary)
        return OutlinesCoreLogitsProcessor(index, self.tensor_library_name, regex)

    def get_cfg_logits_processor(self, grammar):
        raise NotImplementedError(
            "Outlines Core does not support context-free grammar."
        )

    @staticmethod
    def create_outlines_core_vocabulary(
        vocab: Dict[str, int],
        eos_token_id: int,
        eos_token: str,
        token_to_str: Callable[[str], str],
    ) -> Vocabulary:
        """Create an Outlines Core Vocabulary instance.

        Parameters
        ----------
        vocab: Dict[str, int]
            The vocabulary to create an Outlines Core vocabulary from.
        eos_token_id: int
            The EOS token ID.
        eos_token: str
            The EOS token.
        token_to_str: Callable[[str], str]
            The function to convert a token to a string.

        Returns
        -------
        Vocabulary
            The Outlines Core Vocabulary instance.

        """
        formatted_vocab: Dict[str, List[int]] = {}
        for token, token_id in vocab.items():
            # This step is necessary to transform special tokens into their
            # string representation, in particular for spacing. We need those
            # string representations as outlines core first builds an FSM from
            # the regex provided that only contains regular strings.
            token_as_str = token_to_str(token)
            formatted_vocab.setdefault(token_as_str, []).append(token_id)
        eos_token_as_str = token_to_str(eos_token)
        formatted_vocab.pop(eos_token_as_str, None)
        return Vocabulary(eos_token_id, formatted_vocab)
