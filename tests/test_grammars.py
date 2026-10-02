"""Tests for grammar preprocessing and reference-cycle diagnostics."""

from pathlib import Path

import pytest

from outlines import grammars

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def direct_cycle_grammar():
    # "a" only ever calls "b" and vice versa: neither rule can terminate.
    return "start: a\na: b\nb: a\n"


@pytest.fixture
def direct_self_cycle_grammar():
    return "start: a\na: a\n"


@pytest.fixture
def indirect_cycle_grammar():
    return "start: a\na: b\nb: c\nc: a\n"


@pytest.fixture
def legal_left_recursion_grammar():
    # The bundled arithmetic grammar is left recursive and must be accepted.
    return (
        "?start: expr\n"
        'expr: expr "+" term | expr "-" term | term\n'
        "term: NUMBER\n"
        "%import common.NUMBER\n"
    )


@pytest.fixture
def legal_right_recursion_grammar():
    return 'start: list\nlist: NUMBER "," list | NUMBER\n%import common.NUMBER\n'


@pytest.fixture
def legal_nested_recursion_grammar():
    # Mutual recursion where every rule has a terminal base case.
    return 'start: a\na: "x" b | "y"\nb: "z" a | "w"\n'


@pytest.fixture
def missing_rule_reference_grammar():
    return 'start: a\na: b | "x"\n'


@pytest.fixture
def missing_terminal_reference_grammar():
    return "start: a\na: MISSING_TERMINAL\n"


# ---------------------------------------------------------------------------
# Direct cycle
# ---------------------------------------------------------------------------


def test_find_direct_cycle(direct_cycle_grammar):
    cycle = grammars.find_grammar_reference_cycle(direct_cycle_grammar)
    assert cycle == ["a", "b"]


def test_find_direct_self_cycle(direct_self_cycle_grammar):
    cycle = grammars.find_grammar_reference_cycle(direct_self_cycle_grammar)
    assert cycle == ["a"]


def test_preprocess_reports_readable_direct_cycle(direct_cycle_grammar):
    with pytest.raises(grammars.GrammarReferenceCycleError) as exc_info:
        grammars.preprocess_grammar(direct_cycle_grammar)
    assert exc_info.value.cycle == ["a", "b"]
    message = str(exc_info.value)
    # The shortest readable path repeats the first node at the end.
    assert "a -> b -> a" in message


# ---------------------------------------------------------------------------
# Indirect cycle
# ---------------------------------------------------------------------------


def test_find_indirect_cycle(indirect_cycle_grammar):
    cycle = grammars.find_grammar_reference_cycle(indirect_cycle_grammar)
    assert cycle == ["a", "b", "c"]
    assert grammars.cycle_path_to_str(cycle) == "a -> b -> c -> a"


def test_preprocess_reports_indirect_cycle(indirect_cycle_grammar):
    with pytest.raises(grammars.GrammarReferenceCycleError) as exc_info:
        grammars.preprocess_grammar(indirect_cycle_grammar)
    assert exc_info.value.cycle == ["a", "b", "c"]
    assert "a -> b -> c -> a" in str(exc_info.value)


def test_shortest_cycle_is_reported():
    # Two disjoint cycles: the 2-node cycle must win over the 3-node one.
    text = "start: z\nz: zz\nzz: z\na: b\nb: c\nc: a\n"
    cycle = grammars.find_grammar_reference_cycle(text)
    assert cycle == ["z", "zz"]


def test_indirect_cycle_with_base_case_is_legal():
    text = 'start: a\na: b\nb: c | "x"\nc: a\n'
    assert grammars.find_grammar_reference_cycle(text) is None
    grammars.preprocess_grammar(text)


# ---------------------------------------------------------------------------
# Legal recursion must not be flagged
# ---------------------------------------------------------------------------


def test_legal_left_recursion_accepted(legal_left_recursion_grammar):
    analysis = grammars.analyze_grammar_references(legal_left_recursion_grammar)
    assert analysis.cycle is None
    assert analysis.missing == {}
    grammars.preprocess_grammar(legal_left_recursion_grammar)


def test_legal_right_recursion_accepted(legal_right_recursion_grammar):
    analysis = grammars.analyze_grammar_references(legal_right_recursion_grammar)
    assert analysis.cycle is None
    assert analysis.missing == {}
    grammars.preprocess_grammar(legal_right_recursion_grammar)


def test_legal_nested_recursion_accepted(legal_nested_recursion_grammar):
    analysis = grammars.analyze_grammar_references(legal_nested_recursion_grammar)
    assert analysis.cycle is None
    assert analysis.missing == {}
    grammars.preprocess_grammar(legal_nested_recursion_grammar)


def test_bundled_grammars_are_recursion_safe():
    # Regression guard: the bundled grammars use recursion and are loaded
    # with validation enabled at import time.
    assert grammars.arithmetic
    assert grammars.json
    grammars.preprocess_grammar(grammars.arithmetic)
    grammars.preprocess_grammar(grammars.json)


def test_optional_and_star_recursion_are_legal():
    # "[a]" and "(a)*" derive the empty string, so the terminal stays
    # reachable (no unguarded cycle).
    grammars.preprocess_grammar('start: a\na: [a] "x"\n')
    grammars.preprocess_grammar('start: a\na: (a)* "x"\n')
    grammars.preprocess_grammar('start: a\na: a? "x"\n')


# ---------------------------------------------------------------------------
# Missing references are reported separately
# ---------------------------------------------------------------------------


def test_missing_rule_reference(missing_rule_reference_grammar):
    analysis = grammars.analyze_grammar_references(missing_rule_reference_grammar)
    assert analysis.missing == {"b": "a"}
    assert analysis.cycle is None
    with pytest.raises(grammars.GrammarReferenceNotFoundError) as exc_info:
        grammars.preprocess_grammar(missing_rule_reference_grammar)
    assert exc_info.value.missing == {"b": "a"}
    assert "b" in str(exc_info.value)
    assert "a" in str(exc_info.value)


def test_missing_terminal_reference(missing_terminal_reference_grammar):
    with pytest.raises(grammars.GrammarReferenceNotFoundError) as exc_info:
        grammars.preprocess_grammar(missing_terminal_reference_grammar)
    assert exc_info.value.missing == {"MISSING_TERMINAL": "a"}


def test_missing_reference_takes_precedence_over_cycle():
    # "b" is undefined; even though the call graph looks cyclic on paper,
    # missing references are diagnosed first and as their own error type.
    text = "start: a\na: b\nb: c\n"
    with pytest.raises(grammars.GrammarReferenceNotFoundError):
        grammars.preprocess_grammar(text)


def test_declared_and_imported_symbols_are_not_missing():
    grammar = (
        "%declare external\n"
        "start: a\n"
        "a: external | WS_INLINE\n"
        "%import common.WS_INLINE\n"
    )
    grammars.preprocess_grammar(grammar)


# ---------------------------------------------------------------------------
# read_grammar integration
# ---------------------------------------------------------------------------


def test_read_grammar_validates_by_default(tmp_path):
    (tmp_path / "bad.lark").write_text("start: a\na: b\nb: a\n")
    with pytest.raises(grammars.GrammarReferenceCycleError):
        grammars.read_grammar("bad.lark", base_grammar_path=Path(tmp_path))


def test_read_grammar_can_skip_validation(tmp_path):
    (tmp_path / "bad.lark").write_text("start: a\na: b\nb: a\n")
    contents = grammars.read_grammar(
        "bad.lark", base_grammar_path=Path(tmp_path), validate=False
    )
    assert "a: b" in contents


def test_preprocess_returns_text_unchanged():
    text = 'start: a\na: "x"\n'
    assert grammars.preprocess_grammar(text) == text


def test_error_hierarchy():
    assert issubclass(
        grammars.GrammarReferenceCycleError,
        grammars.GrammarPreprocessingError,
    )
    assert issubclass(
        grammars.GrammarReferenceNotFoundError,
        grammars.GrammarPreprocessingError,
    )
