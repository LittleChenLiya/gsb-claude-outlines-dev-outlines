"""Tests for grammar preprocessing: reference-cycle localization.

These tests verify the four acceptance fixtures:

* **Direct cycle** -- a rule that references itself (self-loop) or two rules
  that reference each other directly.
* **Indirect cycle** -- three or more rules that form a reference chain
  looping back to the start.
* **Legal recursion** -- recursive rules that *do* have a terminal exit (a
  productive base case) must not be reported.
* **Missing reference** -- a rule that references a rule or terminal which is
  not defined is reported separately from cycles.

"""

import pytest

from outlines import grammars
from outlines.grammars import (
    GrammarReferenceError,
    _find_productive_rules,
    _find_shortest_cycle,
    _parse_grammar,
    preprocess_grammar,
    read_grammar,
)


class TestDirectCycle:
    """A rule references itself, or two rules reference each other directly."""

    def test_self_loop(self):
        grammar = "start: start"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="self.lark")
        message = str(exc_info.value)
        assert "Non-productive reference cycle" in message
        assert "start -> start" in message

    def test_mutual_direct(self):
        grammar = "start: a\na: start"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="mutual.lark")
        message = str(exc_info.value)
        assert "Non-productive reference cycle" in message
        assert "start -> a -> start" in message


class TestIndirectCycle:
    """Three or more rules form a reference chain looping back to the start."""

    def test_three_rule_cycle(self):
        grammar = "start: a\na: b\nb: c\nc: start"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="indirect.lark")
        message = str(exc_info.value)
        assert "Non-productive reference cycle" in message
        assert "start -> a -> b -> c -> start" in message

    def test_four_rule_cycle(self):
        grammar = "a: b\nb: c\nc: d\nd: a"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="four.lark")
        message = str(exc_info.value)
        assert "a -> b -> c -> d -> a" in message


class TestLegalRecursion:
    """Recursive rules with a productive base case are legal."""

    def test_self_recursion_with_base_case(self):
        grammar = 'start: "a" start | "b"'
        result = preprocess_grammar(grammar, source="legal.lark")
        assert result == grammar

    def test_mutual_recursion_with_base_case(self):
        grammar = 'start: a\na: "x" | b\nb: "y" | a'
        result = preprocess_grammar(grammar, source="mutual_legal.lark")
        assert result == grammar

    def test_cycle_with_productive_exit(self):
        grammar = 'start: a | "x"\na: b | "y"\nb: a'
        result = preprocess_grammar(grammar, source="exit.lark")
        assert result == grammar

    def test_arithmetic_grammar_recursion(self):
        result = preprocess_grammar(grammars.arithmetic, source="arithmetic.lark")
        assert result == grammars.arithmetic

    def test_json_grammar_recursion(self):
        result = preprocess_grammar(grammars.json, source="json.lark")
        assert result == grammars.json


class TestMissingReference:
    """References to undefined rules or terminals are reported separately."""

    def test_missing_rule(self):
        grammar = "start: undefined_rule"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="missing.lark")
        message = str(exc_info.value)
        assert "Missing references" in message
        assert "undefined_rule" in message
        assert "undefined rule" in message

    def test_missing_terminal(self):
        grammar = "start: UNDEFINED_TERM"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="term.lark")
        message = str(exc_info.value)
        assert "Missing references" in message
        assert "UNDEFINED_TERM" in message
        assert "undefined terminal" in message

    def test_missing_and_cycle_reported_separately(self):
        grammar = "start: missing\na: b\nb: a"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="both.lark")
        message = str(exc_info.value)
        assert "Missing references" in message
        assert "missing" in message
        assert "Non-productive reference cycle" in message
        assert "a -> b -> a" in message


class TestShortestCycle:
    """The shortest cycle is returned when multiple cycles exist."""

    def test_self_loop_is_shortest(self):
        grammar = "start: a\na: b | a\nb: a"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="shortest.lark")
        message = str(exc_info.value)
        assert "a -> a" in message

    def test_shortest_among_indirect_cycles(self):
        grammar = "a: b | c\nb: a\nc: d\nd: a"
        with pytest.raises(GrammarReferenceError) as exc_info:
            preprocess_grammar(grammar, source="shortest2.lark")
        message = str(exc_info.value)
        assert "a -> b -> a" in message


class TestProductivity:
    """Productive rules are those that can derive a string of terminals."""

    def test_productive_rules(self):
        grammar = 'start: a | "x"\na: b\nb: a\nc: c'
        rules, _terminals = _parse_grammar(grammar)
        productive = _find_productive_rules(rules)
        assert productive == {"start"}

    def test_cycle_among_non_productive_only(self):
        grammar = 'start: a | "x"\na: b\nb: a\nc: c'
        rules, _terminals = _parse_grammar(grammar)
        productive = _find_productive_rules(rules)
        cycle = _find_shortest_cycle(rules, productive)
        assert cycle == ["c", "c"]


class TestReadGrammar:
    """read_grammar validates by default."""

    def test_valid_grammar(self, tmp_path):
        grammar_file = tmp_path / "valid.lark"
        grammar_file.write_text('start: "a" start | "b"')
        result = read_grammar("valid.lark", base_grammar_path=tmp_path)
        assert result == 'start: "a" start | "b"'

    def test_invalid_grammar_raises(self, tmp_path):
        grammar_file = tmp_path / "invalid.lark"
        grammar_file.write_text("start: start")
        with pytest.raises(GrammarReferenceError):
            read_grammar("invalid.lark", base_grammar_path=tmp_path)

    def test_validation_can_be_disabled(self, tmp_path):
        grammar_file = tmp_path / "skip.lark"
        grammar_file.write_text("start: start")
        result = read_grammar("skip.lark", base_grammar_path=tmp_path, validate=False)
        assert result == "start: start"

    def test_builtin_grammars_load(self):
        assert isinstance(grammars.arithmetic, str)
        assert isinstance(grammars.json, str)
        assert len(grammars.arithmetic) > 0
        assert len(grammars.json) > 0


class TestEdgeCases:
    """Empty grammars and grammars without rules are valid."""

    def test_empty_grammar(self):
        result = preprocess_grammar("", source="empty.lark")
        assert result == ""

    def test_terminal_only_grammar(self):
        grammar = 'DIGIT: "0".."9"\nstart: DIGIT'
        result = preprocess_grammar(grammar, source="term_only.lark")
        assert result == grammar

    def test_grammar_with_aliases(self):
        grammar = 'start: "a" -> a_alias | "b" -> b_alias'
        result = preprocess_grammar(grammar, source="aliases.lark")
        assert result == grammar
