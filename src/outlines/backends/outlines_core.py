"""Backend class for Outlines Core."""

import hashlib
import json
from typing import Callable, Dict, List, Optional

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


#: Version of the state snapshot format produced by
#: :meth:`OutlinesCoreLogitsProcessor.snapshot`.
SNAPSHOT_VERSION = 1


def _vocabulary_fingerprint(
    formatted_vocab: Dict[str, List[int]], eos_token_id: int
) -> str:
    """Compute a stable fingerprint of an Outlines Core vocabulary.

    The fingerprint only depends on the token strings and token ids that
    make up the vocabulary, so two backends built from the same model share
    the same fingerprint.

    Parameters
    ----------
    formatted_vocab: Dict[str, List[int]]
        The formatted vocabulary passed to the Outlines Core ``Vocabulary``.
    eos_token_id: int
        The id of the EOS token.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest.

    """
    items = sorted(
        (token, sorted(int(token_id) for token_id in token_ids))
        for token, token_ids in formatted_vocab.items()
    )
    payload = json.dumps(
        {"eos_token_id": int(eos_token_id), "tokens": items},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _grammar_fingerprint(grammar: str, vocabulary_fingerprint: str) -> str:
    """Compute a stable fingerprint of a grammar.

    The grammar is identified by its regex and the fingerprint of the
    vocabulary it is compiled against. Two processors built for the same
    grammar share the same fingerprint, so a snapshot can only be restored
    on a processor built for the exact same grammar.

    Parameters
    ----------
    grammar: str
        The regex that defines the grammar.
    vocabulary_fingerprint: str
        The fingerprint of the vocabulary the grammar is compiled against.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest.

    """
    payload = json.dumps(
        {"grammar": grammar, "vocabulary": vocabulary_fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OutlinesCoreLogitsProcessor(OutlinesLogitsProcessor):
    """Logits processor for Outlines Core."""

    def __init__(
        self,
        index: Index,
        tensor_library_name: str,
        grammar: Optional[str] = None,
        vocabulary_fingerprint: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        index: Index
            The Outlines Core `Index` instance to use to create the Outlines
            Core `Guide` instances that will be used to bias the logits
        tensor_library_name: str
            The tensor library name to use for the logits processor.
        grammar: Optional[str]
            The regex that defines the grammar the index was built for. It is
            used to identify the grammar when restoring a state snapshot. It
            can be omitted if the snapshot feature is not used.
        vocabulary_fingerprint: Optional[str]
            The fingerprint of the vocabulary the grammar was compiled
            against. It is used together with ``grammar`` to identify the
            grammar when restoring a state snapshot. It can be omitted if the
            snapshot feature is not used.

        """
        self.index = index
        self.tensor_library_name = tensor_library_name
        self.is_first_token = True
        self.grammar = grammar
        if grammar is not None and vocabulary_fingerprint is not None:
            self.grammar_hash = _grammar_fingerprint(grammar, vocabulary_fingerprint)
        else:
            self.grammar_hash = None
        self._guides: Optional[List[Guide]] = None
        self._bitmasks: Optional[List] = None
        self._consumed_tokens: List[List[int]] = []
        self._pending_restore: Optional[List[List[int]]] = None
        super().__init__(tensor_library_name)

    def reset(self) -> None:
        """Reset the logits processor for a new generation.

        This clears the guides, the bitmasks and the consumed tokens, but it
        intentionally preserves a pending restore staged by
        :meth:`restore`: the first call to :meth:`process_logits` will replay
        the restored token prefix instead of starting from scratch.

        """
        self.is_first_token = True
        self._guides = None
        self._bitmasks = None
        self._consumed_tokens = []

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

        if self._pending_restore is not None:
            # Replay the token prefix staged by `restore` so the guides resume
            # from the exact state they had when the snapshot was taken.
            if len(self._pending_restore) != batch_size:
                raise ValueError(
                    "Cannot restore snapshot: the batch size of the logits "
                    f"({batch_size}) does not match the batch size of the "
                    f"snapshot ({len(self._pending_restore)})."
                )
            self._guides = []
            self._consumed_tokens = []
            for token_list in self._pending_restore:
                guide = Guide(self.index)
                consumed: List[int] = []
                for token_id in token_list:
                    guide.advance(token_id=token_id, return_tokens=False)
                    consumed.append(int(token_id))
                self._guides.append(guide)
                self._consumed_tokens.append(consumed)
            self._pending_restore = None
        else:
            self._guides = [Guide(self.index) for _ in range(batch_size)]
            self._consumed_tokens = [[] for _ in range(batch_size)]

        self._bitmasks = [
            self.allocate_token_bitmask(vocab_size) for _ in range(batch_size)
        ]

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
        else:
            if self._bitmasks is None:
                self._bitmasks = [
                    self.allocate_token_bitmask(vocab_size)
                    for _ in range(batch_size)
                ]
            for i in range(batch_size):
                last_token_id = self.tensor_adapter.to_scalar(input_ids[i][-1])  # type: ignore
                # This circumvents issue #227 in outlines_core
                # Ideally, we would be able to advance all the times as the final
                # state would accept the eos token leading to itself
                if not self._guides[i].is_finished() or self._guides[i].accepts_tokens(
                    [last_token_id]
                ):
                    self._guides[i].advance(token_id=last_token_id, return_tokens=False)
                    self._consumed_tokens[i].append(int(last_token_id))

        return self.bias_logits(batch_size, logits)

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the current constraint state.

        The snapshot captures the identity of the grammar the processor was
        built for and the token prefix consumed by each guide. It can be
        passed to :meth:`restore` on another processor built for the same
        grammar to resume incremental parsing from the exact same state.

        Returns
        -------
        dict
            A JSON-serializable dictionary with the following keys:

            - ``version``: the snapshot format version.
            - ``grammar``: the regex that defines the grammar.
            - ``grammar_hash``: the fingerprint of the grammar.
            - ``tokens``: the list of consumed token ids, one per guide.

        """
        if self._guides is not None:
            tokens = [
                list(self._consumed_tokens[i]) for i in range(len(self._guides))
            ]
        elif self._pending_restore is not None:
            tokens = [list(token_list) for token_list in self._pending_restore]
        else:
            tokens = []
        return {
            "version": SNAPSHOT_VERSION,
            "grammar": self.grammar,
            "grammar_hash": self.grammar_hash,
            "tokens": tokens,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore the constraint state from a snapshot.

        The snapshot must have been produced by :meth:`snapshot` on a
        processor built for the exact same grammar (same regex and same
        vocabulary). Otherwise a ``ValueError`` is raised and the state of
        this processor is left untouched.

        The restored state is staged and replayed on the next call to
        :meth:`process_logits`, so the first call after a restore does not
        advance the guides: it only biases the logits from the restored
        state.

        Parameters
        ----------
        snapshot: dict
            A snapshot produced by :meth:`snapshot`.

        Raises
        ------
        ValueError
            If the snapshot is not a valid snapshot, if its version is not
            supported, or if the grammar it was built for does not match the
            grammar of this processor.

        """
        if not isinstance(snapshot, dict):
            raise ValueError(
                "Snapshot must be a dictionary produced by the snapshot() method."
            )
        if snapshot.get("version") != SNAPSHOT_VERSION:
            raise ValueError(
                f"Unsupported snapshot version: {snapshot.get('version')!r}. "
                f"Expected version {SNAPSHOT_VERSION}."
            )
        snapshot_hash = snapshot.get("grammar_hash")
        if self.grammar_hash is not None:
            if snapshot_hash != self.grammar_hash:
                raise ValueError(
                    "Cannot restore snapshot: the grammar it was built for "
                    "does not match this processor's grammar. Snapshots can "
                    "only be restored on a processor built for the exact same "
                    "grammar."
                )
        elif snapshot_hash is not None:
            raise ValueError(
                "Cannot restore snapshot: it carries a grammar identity, but "
                "this processor was built without one."
            )
        tokens = snapshot.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token_list, list) for token_list in tokens
        ):
            raise ValueError(
                "Invalid snapshot: 'tokens' must be a list of token id lists."
            )
        # Stage the restore. `reset` clears the guides, the bitmasks and the
        # consumed tokens, but it preserves a pending restore, so the first
        # call to `process_logits` replays the token prefix instead of
        # starting from scratch.
        self.reset()
        if len(tokens) > 0:
            self._pending_restore = [
                [int(token_id) for token_id in token_list] for token_list in tokens
            ]
        else:
            self._pending_restore = None


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
        self.vocabulary, self.vocabulary_fingerprint = (
            self.create_outlines_core_vocabulary(
                vocabulary, eos_token_id, eos_token, token_to_str
            )
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
        return OutlinesCoreLogitsProcessor(
            index,
            self.tensor_library_name,
            grammar=regex,
            vocabulary_fingerprint=self.vocabulary_fingerprint,
        )

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
    ) -> tuple[Vocabulary, str]:
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
        Tuple[Vocabulary, str]
            The Outlines Core Vocabulary instance and a stable fingerprint of
            the formatted vocabulary.

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
        vocabulary = Vocabulary(eos_token_id, formatted_vocab)
        fingerprint = _vocabulary_fingerprint(formatted_vocab, eos_token_id)
        return vocabulary, fingerprint
