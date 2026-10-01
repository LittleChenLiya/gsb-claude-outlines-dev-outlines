"""Encapsulate a model and an output type into a reusable object."""

import json
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from outlines.backends import (
    get_cfg_logits_processor,
    get_json_schema_logits_processor,
    get_regex_logits_processor,
)
from outlines.backends.base import LogitsProcessorType
from outlines.exceptions import StructuredGenerationError
from outlines.models import (
    AsyncBlackBoxModel,
    BlackBoxModel,
    SteerableModel,
)
from outlines.models.base import AsyncModel, Model
from outlines.types import CFG, JsonSchema
from outlines.types.dsl import python_types_to_terms, to_regex
from outlines.types.utils import is_native_dict


def _is_valid_json_prefix(s: str) -> Tuple[bool, int]:
    """Check whether *s* is a valid prefix of a JSON value.

    A string is a valid prefix if it is empty, a complete JSON value, or can be
    extended to a complete JSON value by appending more characters.

    Parameters
    ----------
    s
        The string to check.

    Returns
    -------
    Tuple[bool, int]
        ``(True, len(s))`` if *s* is a valid prefix, ``(False, pos)`` otherwise
        where *pos* is the position at which the first invalid character was
        encountered.

    """
    if not s:
        return (True, 0)

    try:
        json.loads(s)
        return (True, len(s))
    except json.JSONDecodeError as exc:
        # An unterminated string literal (e.g. '"abc') is not closed yet, but
        # it can still be completed by appending more characters, so it is a
        # valid prefix.
        if exc.msg.startswith("Unterminated string"):
            return (True, len(s))
        # An error at or beyond the end of the input means the value is simply
        # incomplete (e.g. '{"a": 1'), not invalid.
        if exc.pos >= len(s):
            return (True, len(s))
        return (False, exc.pos)


def _verify_json_stream(stream: Iterator[str]) -> Iterator[str]:
    """Wrap *stream* so that the accumulated output always forms a valid JSON
    prefix.

    Yields each chunk as it is received. Raises
    :class:`~outlines.exceptions.StructuredGenerationError` if the accumulated
    output ceases to be a valid prefix (the model emitted an invalid character)
    or if the stream ends before the output is a complete JSON value.

    Parameters
    ----------
    stream
        The raw stream of text chunks from the model.

    Yields
    ------
    str
        The chunks from *stream*, unchanged.

    """
    accumulated = ""
    for chunk in stream:
        accumulated += chunk
        is_valid, pos = _is_valid_json_prefix(accumulated)
        if not is_valid:
            raise StructuredGenerationError(
                verified_prefix=accumulated[:pos],
                failure_position=pos,
                reason="invalid",
            )
        yield chunk

    if accumulated:
        try:
            json.loads(accumulated)
        except json.JSONDecodeError:
            is_valid, pos = _is_valid_json_prefix(accumulated)
            raise StructuredGenerationError(
                verified_prefix=accumulated[:pos],
                failure_position=pos,
                reason="incomplete",
            )


async def _verify_json_stream_async(
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Async counterpart of :func:`_verify_json_stream`."""
    accumulated = ""
    async for chunk in stream:
        accumulated += chunk
        is_valid, pos = _is_valid_json_prefix(accumulated)
        if not is_valid:
            raise StructuredGenerationError(
                verified_prefix=accumulated[:pos],
                failure_position=pos,
                reason="invalid",
            )
        yield chunk

    if accumulated:
        try:
            json.loads(accumulated)
        except json.JSONDecodeError:
            is_valid, pos = _is_valid_json_prefix(accumulated)
            raise StructuredGenerationError(
                verified_prefix=accumulated[:pos],
                failure_position=pos,
                reason="incomplete",
            )


class BlackBoxGenerator:
    """Synchronous generator for which we don't control constrained
    generation.

    The output type provided is not compiled into a logits processor, but is
    instead directly passed on to the model.

    """
    output_type: Optional[Any]

    def __init__(self, model: BlackBoxModel, output_type: Optional[Any]):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type that will be used to constrain the generation.

        """
        self.model = model
        self.output_type = output_type

    def _is_json_output_type(self) -> bool:
        """Return ``True`` if the output type is a JSON type."""
        if self.output_type is None:
            return False
        return JsonSchema.is_json_schema(self.output_type) or is_native_dict(
            self.output_type
        )

    def _verify_json_result(self, result: Any) -> None:
        """Verify that *result* is a complete JSON value.

        Raises
        ------
        StructuredGenerationError
            If *result* is not a complete JSON value.

        """
        if isinstance(result, list):
            for item in result:
                self._verify_json_result(item)
            return

        is_valid, pos = _is_valid_json_prefix(result)
        if not is_valid:
            raise StructuredGenerationError(
                verified_prefix=result[:pos],
                failure_position=pos,
                reason="invalid",
            )
        try:
            json.loads(result)
        except json.JSONDecodeError:
            raise StructuredGenerationError(
                verified_prefix=result[:pos],
                failure_position=pos,
                reason="incomplete",
            )

    def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        result = self.model.generate(
            prompt, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            self._verify_json_result(result)
        return result

    def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        results = self.model.generate_batch(
            prompts, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            for result in results:
                self._verify_json_result(result)
        return results

    def stream(self, prompt: Any, **inference_kwargs) -> Iterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        stream = self.model.generate_stream(
            prompt, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            return _verify_json_stream(stream)
        return stream


class AsyncBlackBoxGenerator:
    """Asynchronous generator for which we don't control constrained
    generation.

    The output type provided is not compiled into a logits processor, but is
    instead directly passed on to the model.

    """
    output_type: Optional[Any]

    def __init__(self, model: AsyncBlackBoxModel, output_type: Optional[Any]):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type that will be used to constrain the generation.

        """
        self.model = model
        self.output_type = output_type

    def _is_json_output_type(self) -> bool:
        """Return ``True`` if the output type is a JSON type."""
        if self.output_type is None:
            return False
        return JsonSchema.is_json_schema(self.output_type) or is_native_dict(
            self.output_type
        )

    def _verify_json_result(self, result: Any) -> None:
        """Verify that *result* is a complete JSON value.

        Raises
        ------
        StructuredGenerationError
            If *result* is not a complete JSON value.

        """
        if isinstance(result, list):
            for item in result:
                self._verify_json_result(item)
            return

        is_valid, pos = _is_valid_json_prefix(result)
        if not is_valid:
            raise StructuredGenerationError(
                verified_prefix=result[:pos],
                failure_position=pos,
                reason="invalid",
            )
        try:
            json.loads(result)
        except json.JSONDecodeError:
            raise StructuredGenerationError(
                verified_prefix=result[:pos],
                failure_position=pos,
                reason="incomplete",
            )

    async def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        result = await self.model.generate(
            prompt, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            self._verify_json_result(result)
        return result

    async def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        results = await self.model.generate_batch(
            prompts, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            for result in results:
                self._verify_json_result(result)
        return results

    async def stream(self, prompt: Any, **inference_kwargs) -> AsyncIterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        stream = self.model.generate_stream(  # pragma: no cover
            prompt, self.output_type, **inference_kwargs
        )
        if self._is_json_output_type():
            async for chunk in _verify_json_stream_async(stream):
                yield chunk
        else:
            async for chunk in stream:
                yield chunk


class SteerableGenerator:
    """Represents a generator for which we control constrained generation.

    The generator is responsible for building and storing the logits processor
    (which can be quite expensive to build), and then passing it to the model
    when the generator is called.

    The argument defining constrained generation can be of 2 types associated
    to different methods to create an instance of the generator:
    - `output_type` (through `__init__`): an output type as defined in the
      `outlines.types` module
    - `processor` (through `from_processor`): an already built logits processor
       as defined in the `outlines.processors` module

    The 2 parameters are mutually exclusive.

    """
    logits_processor: Optional[LogitsProcessorType]

    def __init__(
        self,
        model: SteerableModel,
        output_type: Optional[Any],
        backend_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model
            An instance of an Outlines model.
        output_type
            The output type expressed as a Python type
        backend_name
            The name of the backend to use to create the logits processor.

        """
        self.model = model
        if output_type is None:
            self.logits_processor = None
        else:
            term = python_types_to_terms(output_type)
            if isinstance(term, CFG):
                cfg_string = term.definition
                self.logits_processor = get_cfg_logits_processor(
                    backend_name,
                    model,
                    cfg_string,
                )
            elif isinstance(term, JsonSchema):
                self.logits_processor = get_json_schema_logits_processor(
                    backend_name,
                    model,
                    term.schema,
                    term.whitespace_pattern,
                )
            else:
                regex_string = to_regex(term)
                self.logits_processor = get_regex_logits_processor(
                    backend_name,
                    model,
                    regex_string,
                )

    @classmethod
    def from_processor(
        cls, model: SteerableModel, processor: LogitsProcessorType
    ):
        """Create a generator from a logits processor.

        Parameters
        ----------
        model
            An instance of an Outlines model.
        processor
            An instance of a logits processor.

        """
        instance = cls.__new__(cls)
        instance.model = model
        instance.logits_processor = processor

        return instance

    def __call__(self, prompt: Any, **inference_kwargs) -> Any:
        """Generate a response from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        if self.logits_processor is not None:
            self.logits_processor.reset()
        return self.model.generate(
            prompt, self.logits_processor, **inference_kwargs
        )

    def batch(self, prompts: List[Any], **inference_kwargs) -> List[Any]:
        """Generate a batch of responses from the model.

        Parameters
        ----------
        prompts
            The list of prompts to use to generate a batch of responses.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        List[Any]
            The list of responses generated by the model.

        """
        if self.logits_processor is not None:
            self.logits_processor.reset()
        return self.model.generate_batch(
            prompts, self.logits_processor, **inference_kwargs
        )

    def stream(self, prompt: Any, **inference_kwargs) -> Iterator[Any]:
        """Generate a stream of responses from the model.

        Parameters
        ----------
        prompt
            The prompt to use to generate a response.
        **inference_kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The response generated by the model.

        """
        if self.logits_processor is not None:
            self.logits_processor.reset()
        return self.model.generate_stream(
            prompt, self.logits_processor, **inference_kwargs
        )


def Generator(
    model: Union[Model, AsyncModel],
    output_type: Optional[Any] = None,
    backend: Optional[str] = None,
    *,
    processor: Optional[LogitsProcessorType] = None,
) -> Union[SteerableGenerator, BlackBoxGenerator, AsyncBlackBoxGenerator]:
    """Create a generator for the given model and output parameters.

    The 2 parameters output_type and processor are mutually exclusive. The
    parameters processor is only supported for SteerableModel instances
    (typically local models) and is intended to be only used by advanced users.

    Parameters
    ----------
    model
        An instance of an Outlines model.
    output_type
        The output type expressed as a Python type or a type defined in the
        outlines.types.dsl module.
    backend
        The name of the backend to use to create the logits processor. Only
        used for steerable models if there is an output type and `processor` is
        not provided.
    processor
        An instance of a logits processor.

    Returns
    -------
    Union[SteerableGenerator, BlackBoxGenerator, AsyncBlackBoxGenerator]
        A generator instance.

    """
    provided_output_params = sum(
        param is not None
        for param in [output_type, processor]
    )
    if provided_output_params > 1:
        raise ValueError(
            "At most one of output_type or processor can be provided"
        )

    if isinstance(model, SteerableModel): # type: ignore
        if processor is not None:
            return SteerableGenerator.from_processor(model, processor) # type: ignore
        else:
            return SteerableGenerator(model, output_type, backend) # type: ignore
    else:
        if processor is not None:
            raise NotImplementedError(
                "This model does not support logits processors"
            )
        if isinstance(model, AsyncBlackBoxModel): # type: ignore
            return AsyncBlackBoxGenerator(model, output_type) # type: ignore
        elif isinstance(model, BlackBoxModel): # type: ignore
            return BlackBoxGenerator(model, output_type) # type: ignore
        else:
            raise ValueError(
                "The model argument must be an instance of "
                "SteerableModel, BlackBoxModel or AsyncBlackBoxModel"
            )
