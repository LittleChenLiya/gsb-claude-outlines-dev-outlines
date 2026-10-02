"""Common Lark grammars and grammar preprocessing.

Besides loading the bundled ``.lark`` grammars, this module preprocesses
grammar text with diagnostics that point at *reference cycles* between rules:

* :class:`GrammarReferenceCycleError` -- at least one rule can never produce a
  finite string, because its alternatives only ever call other rules (no
  terminal token guards the recursion). The error carries the shortest
  readable cycle path, e.g. ``a -> b -> a``.
* :class:`GrammarReferenceNotFoundError` -- a rule body references a name that
  is neither defined, declared nor imported. This is reported separately from
  cycles, the way Lark separates "undefined symbol" from grammar problems.

The preprocessor uses a small self-contained tokenizer/parser for the subset
of the Lark syntax that matters for rule-to-rule references, so it does not
depend on Lark itself at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

GRAMMAR_PATH = Path(__file__).parent / "grammars"

__all__ = [
    "GrammarPreprocessingError",
    "GrammarReferenceAnalysis",
    "GrammarReferenceCycleError",
    "GrammarReferenceNotFoundError",
    "analyze_grammar_references",
    "arithmetic",
    "find_grammar_reference_cycle",
    "json",
    "preprocess_grammar",
    "read_grammar",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GrammarPreprocessingError(ValueError):
    """Base class for grammar preprocessing problems."""


class GrammarReferenceNotFoundError(GrammarPreprocessingError):
    """Raised when a rule references a name that is not defined anywhere."""

    def __init__(self, missing: dict):
        self.missing = dict(missing)
        first_name, first_owner = min(missing.items())
        super().__init__(
            f"Undefined grammar reference '{first_name}' used by rule "
            f"'{first_owner}'. Define it, declare it with %declare or import "
            f"it with %import. All missing references: {sorted(missing)}"
        )


class GrammarReferenceCycleError(GrammarPreprocessingError):
    """Raised when rules reference each other without any terminal guard."""

    def __init__(self, cycle: list):
        self.cycle = list(cycle)
        readable = cycle_path_to_str(cycle)
        super().__init__(
            f"Unguarded reference cycle between grammar rules: {readable}. "
            "Every rule on this path only calls other rules, so none of them "
            "can ever terminate. Add an alternative containing a terminal "
            "token (a base case) to break the cycle."
        )


def cycle_path_to_str(cycle: list) -> str:
    """Render a cycle node list as a readable path (first node repeated)."""

    return " -> ".join([*cycle, cycle[0]])


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


class _NodeKind(str, Enum):
    RULE_REF = "rule_ref"
    TERMINAL_REF = "terminal_ref"
    STRING = "string"
    REGEXP = "regexp"
    RANGE = "range"
    GROUP = "group"


@dataclass
class _Node:
    kind: _NodeKind
    name: str = ""
    children: list = field(default_factory=list)
    # Modifier following the node: "*" (zero or more), "?" (optional),
    # "+" (one or more) or None.
    op: str | None = None
    # Whether a leading "~" prefix filter was applied; such nodes consume a
    # token, so they are treated as terminal guards.
    filtered: bool = False
    # Leading "-" / "~" lookahead; purely declarative, never a call to satisfy.
    predicate: bool = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list:
    """Tokenize Lark grammar text into ``(type, value, position)`` tuples.

    Token types: ``NAME``, ``STRING``, ``REGEXP``, ``OP``, ``DIRECTIVE`` and
    ``NEWLINE`` (one per physical line break; blank lines are dropped by the
    parser). Newlines are kept so a definition header can be recognized as a
    name at the start of a logical line.
    """

    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            tokens.append(("NEWLINE", "\n", i))
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            if j == -1:
                i = n
            else:
                i = j  # keep the newline as a NEWLINE token
            continue
        if c == "#":
            j = text.find("\n", i)
            if j == -1:
                i = n
            else:
                i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                i = n
            else:
                i = j + 2
                # A block comment ending the line still ends the line.
                if i < n and text[i] == "\n":
                    i = j + 2
            continue
        if c == "%":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("DIRECTIVE", text[i:j], i))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("NAME", text[i:j], i))
            i = j
            continue
        if c in "\"'":
            quote = c
            j = i + 1
            triple = text.startswith(quote * 3, i)
            if triple:
                end = text.find(quote * 3, j + 2)
                if end == -1:
                    raise GrammarPreprocessingError(
                        "Unterminated string literal in grammar"
                    )
                j = end + 3
            else:
                while j < n:
                    ch = text[j]
                    if ch == "\\":
                        if j + 1 >= n:
                            raise GrammarPreprocessingError(
                                "Unterminated string literal in grammar"
                            )
                        j += 2
                        continue
                    if ch == quote:
                        break
                    if ch == "\n":
                        raise GrammarPreprocessingError(
                            "Unterminated string literal in grammar"
                        )
                    j += 1
                if j >= n:
                    raise GrammarPreprocessingError(
                        "Unterminated string literal in grammar"
                    )
                j += 1
            tokens.append(("STRING", text[i:j], i))
            i = j
            continue
        if c == "/":
            j = i + 1
            in_class = False
            while j < n:
                ch = text[j]
                if ch == "\\":
                    if j + 1 >= n:
                        raise GrammarPreprocessingError(
                            "Unterminated regexp in grammar"
                        )
                    j += 2
                    continue
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    break
                elif ch == "\n":
                    raise GrammarPreprocessingError("Unterminated regexp in grammar")
                j += 1
            if j >= n:
                raise GrammarPreprocessingError("Unterminated regexp in grammar")
            j += 1
            while j < n and text[j].isalpha():  # regexp flags, e.g. /x/il
                j += 1
            tokens.append(("REGEXP", text[i:j], i))
            i = j
            continue
        # Multi-character operators first ("..=" before "..").
        if text.startswith("..=", i):
            tokens.append(("OP", "..=", i))
            i += 3
            continue
        if text.startswith("..", i):
            tokens.append(("OP", "..", i))
            i += 2
            continue
        tokens.append(("OP", c, i))
        i += 1
    return tokens


# ---------------------------------------------------------------------------
# Parser (rule bodies only -- statement headers are handled structurally)
# ---------------------------------------------------------------------------


class _Parser:
    """Recursive-descent parser producing :class:`_Node` rule bodies."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse_expansions(self):
        """Parse ``alt ("|" alt)*`` until ``)``/``]``/end of tokens."""

        alternatives = [self.parse_expansion()]
        while self._peek() is not None and self._peek()[1] == "|":
            self._next()
            nxt = self._peek()
            if nxt is None or nxt[1] == ")":
                alternatives.append([])  # trailing "|" -> empty alternative
            else:
                alternatives.append(self.parse_expansion())
        self._consume_priority()
        return alternatives

    def _consume_priority(self):
        # A rule may end with an inline priority like ".10".
        if self._peek() is not None and self._peek()[1] == ".":
            self._next()
            while self._peek() is not None and self._peek()[1].isdigit():
                self._next()

    def parse_expansion(self):
        items = []
        while True:
            tok = self._peek()
            if tok is None:
                break
            typ, val = tok[0], tok[1]
            if typ == "NEWLINE" or val in ("|", ")", "]"):
                break
            if typ == "DIRECTIVE":
                break
            items.append(self.parse_value())
        return items

    def parse_value(self):
        filtered = False
        predicate = False
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in ("!", "~", "-"):
                break
            nxt = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
            self._next()
            if tok[1] == "~" and not (nxt is not None and nxt[1] == "!"):
                # "~atom" filters the matched token: it still consumes input.
                filtered = True
            else:
                predicate = True
            # "~!atom" is a negative lookahead filter: consume "!" too.
            if tok[1] == "~" and nxt is not None and nxt[1] == "!":
                self._next()

        tok = self._peek()
        if tok is None:
            raise GrammarPreprocessingError(
                "Unexpected end of grammar while parsing rule body"
            )
        typ, val, _ = tok
        if val in ("(", "["):
            close = ")" if val == "(" else "]"
            implicit_op = None if val == "(" else "?"
            self._next()
            children = self.parse_expansions()
            end = self._peek()
            if end is None or end[1] != close:
                raise GrammarPreprocessingError(f"Unmatched '{val}' in grammar")
            self._next()
            node = _Node(kind=_NodeKind.GROUP, children=children)
            node.op = implicit_op
        elif typ == "NAME":
            self._next()
            if self._peek() is not None and self._peek()[1] == "{":
                self._skip_braces()  # template call arguments
            if val[:1].isupper():
                node = _Node(kind=_NodeKind.TERMINAL_REF, name=val)
            else:
                node = _Node(kind=_NodeKind.RULE_REF, name=val)
        elif typ == "STRING":
            self._next()
            # String "{" / "}" are literal tokens, not template syntax.
            is_brace_literal = val.strip("\"'") in ("{", "}")
            if (
                not is_brace_literal
                and self._peek() is not None
                and self._peek()[1] in ("..", "..=")
            ):
                operator = self._next()[1]
                end = self._next()
                node = _Node(kind=_NodeKind.RANGE, name=f"{val}{operator}{end[1]}")
            else:
                node = _Node(kind=_NodeKind.STRING, name=val)
        elif typ == "REGEXP":
            self._next()
            node = _Node(kind=_NodeKind.REGEXP, name=val)
        elif typ == "OP":
            # A bare operator in a rule body is an anonymous literal token,
            # e.g. 'sliceop: ":" [test]' or 'a: "," b'.
            self._next()
            node = _Node(kind=_NodeKind.STRING, name=val)
        else:
            raise GrammarPreprocessingError(
                f"Unexpected token {val!r} while parsing grammar rule"
            )

        node.filtered = filtered
        node.predicate = predicate
        nxt = self._peek()
        if nxt is not None and nxt[1] in ("?", "*", "+"):
            node.op = self._next()[1]
        # Inline alias: "-> name"
        if (
            len(self.tokens) > self.i + 1
            and self.tokens[self.i][1] == "-"
            and self.tokens[self.i + 1][1] == ">"
        ):
            self._next()
            self._next()
            if self._peek() is not None and self._peek()[0] == "NAME":
                self._next()
        return node

    def _skip_braces(self):
        # The current token is the opening brace.
        depth = 0
        while True:
            tok = self._peek()
            if tok is None:
                raise GrammarPreprocessingError("Unmatched '{' in grammar")
            if tok[1] == "{":
                depth += 1
            elif tok[1] == "}":
                depth -= 1
            self._next()
            if depth == 0:
                return


# ---------------------------------------------------------------------------
# Statement-level parsing
# ---------------------------------------------------------------------------


@dataclass
class _ParsedGrammar:
    rule_defs: dict
    term_defs: dict
    declared: set
    wildcard_modules: set
    explicit_imports: set
    ignored: list
    template_params: set


def _skip_balanced(tokens, i, open_v, close_v):
    depth = 0
    n = len(tokens)
    while i < n:
        if tokens[i][1] == open_v:
            depth += 1
        elif tokens[i][1] == close_v:
            depth -= 1
            i += 1
            if depth == 0:
                return i
        i += 1
    raise GrammarPreprocessingError(f"Unmatched '{open_v}' in grammar")


def _parse_grammar_text(text: str):
    """Parse the top-level structure of a Lark grammar."""

    raw = _tokenize(text)
    rule_defs = {}
    term_defs = {}
    declared = set()
    wildcard_modules = set()
    explicit_imports = set()
    ignored = []
    template_params = set()

    i, n = 0, len(raw)
    while i < n:
        typ, val, _pos = raw[i]
        if typ == "NEWLINE":
            i += 1
            continue
        if typ == "DIRECTIVE":
            directive = val[1:]
            i += 1
            if directive == "import":
                # %import dotted.name [-> ALIAS]
                # %import dotted (NAME, NAME, ...)
                # %import dotted.*  (also accepted for diagnostics)
                path = []
                while i < n and raw[i][0] == "NAME":
                    path.append(raw[i][1])
                    i += 1
                    if i < n and raw[i][1] == ".":
                        i += 1
                        if i < n and raw[i][1] == "*":
                            i += 1
                            wildcard_modules.add(".".join(path))
                            path = []
                            break
                if i < n and raw[i][1] == "(":
                    close_i = _skip_balanced(raw, i, "(", ")")
                    for j in range(i + 1, close_i - 1):
                        if raw[j][0] == "NAME":
                            explicit_imports.add(raw[j][1])
                    i = close_i
                elif path:
                    # Single import: the last path component is the symbol.
                    explicit_imports.add(path[-1])
                # Optional "-> ALIAS".
                if i < n and raw[i][1] == "-" and i + 1 < n and raw[i + 1][1] == ">":
                    i += 2
                    if i < n and raw[i][0] == "NAME":
                        i += 1
                continue
            if directive == "declare":
                while i < n and raw[i][0] != "NEWLINE":
                    if raw[i][0] == "NAME":
                        declared.add(raw[i][1])
                    i += 1
                continue
            if directive == "ignore":
                body_tokens, i = _collect_body(raw, i)
                body_tokens = [t for t in body_tokens if t[0] != "NEWLINE"]
                ignored.extend(_Parser(body_tokens).parse_expansions())
                continue
            # Any other directive (%extends, ...): nothing to track.
            continue

        if typ == "OP" and val in ("!", "?", "_", "@"):
            # Rule/token modifiers: only skip when a header name follows on
            # the same logical line; otherwise leave the token untouched.
            k = i + 1
            while k < n and raw[k][0] == "OP" and raw[k][1] in ("!", "?", "_", "@"):
                k += 1
            if k < n and raw[k][0] == "NAME" and _is_header_at(raw, k):
                i += 1
                continue
            i += 1
            continue

        if typ != "NAME":
            i += 1
            continue

        header_name = val
        i += 1
        # Advance to the ":" of the definition header, skipping template
        # params ("{...}") and the priority (".NUMBER"). Headers never span
        # lines.
        while i < n:
            v = raw[i][1]
            if v == ":":
                break
            if v == "{":
                close_i = _skip_balanced(raw, i, "{", "}")
                # Header template params are placeholders, not references.
                for k in range(i + 1, close_i - 1):
                    if raw[k][0] == "NAME":
                        template_params.add(raw[k][1])
                i = close_i
                continue
            if raw[i][0] == "NEWLINE":
                i = n
                break
            i += 1
        if i >= n:
            break
        i += 1  # consume ":"

        body_tokens, i = _collect_body(raw, i)
        body_tokens = [t for t in body_tokens if t[0] != "NEWLINE"]
        alternatives = _Parser(body_tokens).parse_expansions()
        if header_name[:1].isupper():
            term_defs.setdefault(header_name, []).extend(alternatives)
        else:
            rule_defs.setdefault(header_name, []).extend(alternatives)

    return _ParsedGrammar(
        rule_defs=rule_defs,
        term_defs=term_defs,
        declared=declared,
        wildcard_modules=wildcard_modules,
        explicit_imports=explicit_imports,
        ignored=ignored,
        template_params=template_params,
    )


def _collect_body(tokens, start):
    """Collect a rule body up to the next definition header.

    A body ends at a top-level newline whose next logical line starts a new
    statement (a definition header or a directive), or at a top-level
    directive on the same line. Continuation lines starting with ``|`` are
    part of the body.
    """

    body = []
    i = start
    n = len(tokens)
    depth = 0
    while i < n:
        t, v, _ = tokens[i]
        if v in ("(", "["):
            depth += 1
        elif v in (")", "]"):
            depth = max(0, depth - 1)
        elif v == "{":
            close_i = _skip_balanced(tokens, i, "{", "}")
            body.extend(tokens[i:close_i])
            i = close_i
            continue
        elif depth == 0:
            if t == "DIRECTIVE" and v[1:] in ("import", "declare", "ignore"):
                break
            if t == "NEWLINE" and _statement_starts_after_newline(tokens, i):
                break
            # Continuation line: keep the newline as ordinary whitespace.
        body.append(tokens[i])
        i += 1
    return body, i


def _statement_starts_after_newline(tokens, newline_idx):
    """Whether the first token after *newline_idx* starts a new statement."""

    n = len(tokens)
    j = newline_idx + 1
    while j < n and tokens[j][0] == "NEWLINE":
        j += 1
    if j >= n:
        return False
    t, v = tokens[j][0], tokens[j][1]
    if t == "DIRECTIVE":
        return True
    # Optional rule/token modifiers before the header name.
    while t == "OP" and v in ("!", "?", "_", "@"):
        j += 1
        if j >= n:
            return False
        t, v = tokens[j][0], tokens[j][1]
    if t != "NAME":
        return False  # continuation line (e.g. starts with "|")
    return _is_header_at(tokens, j)


def _is_header_at(tokens, idx):
    """Whether the NAME at *idx* starts a definition (followed by ':').

    Only the remainder of the same logical line is inspected: a header name
    is optionally followed by template params, modifiers and a priority and
    then by ``:``. A name merely referenced inside a rule body never has
    ``:`` later on its own line.
    """

    j = idx + 1
    n = len(tokens)
    while j < n:
        typ, v = tokens[j][0], tokens[j][1]
        if v == ":":
            return True
        if typ == "NEWLINE":
            return False
        if v == "{":
            j = _skip_balanced(tokens, j, "{", "}")
            continue
        j += 1
    return False


# ---------------------------------------------------------------------------
# Reference analysis
# ---------------------------------------------------------------------------


@dataclass
class GrammarReferenceAnalysis:
    """Result of inspecting rule references in a grammar.

    Attributes
    ----------
    rules
        All rule names defined in the grammar.
    missing
        Mapping of missing reference name -> name of the rule using it.
    cycle
        Nodes of the shortest unguarded reference cycle (each node once; the
        first node closes the cycle), or ``None`` when there is no cycle.
    productive
        Rules with at least one alternative that can derive a finite
        terminal-only string. Rules absent from this set never terminate.
    """

    rules: set
    missing: dict
    cycle: list | None
    productive: set


def _iter_nodes(alternatives):
    for alt in alternatives:
        for node in alt:
            yield node
            if node.kind is _NodeKind.GROUP:
                yield from _iter_nodes(node.children)


def _stdlib_terminal_names(module):
    """Terminal names exported by a bundled Lark standard library file."""

    path = GRAMMAR_PATH / (module + ".lark")
    if not path.exists():
        return set()
    parsed = _parse_grammar_text(path.read_text())
    names = set(parsed.term_defs)
    names |= {name for name in parsed.declared if name[:1].isupper()}
    return names


def _terminal_available(name, parsed, known_terminals):
    if name in parsed.term_defs or name in parsed.declared:
        return True
    if known_terminals is not None and name in known_terminals:
        return True
    if name in parsed.explicit_imports:
        return True
    for module in parsed.wildcard_modules:
        if name in _stdlib_terminal_names(module):
            return True
    return False


def analyze_grammar_references(grammar_text, known_terminals=None):
    """Analyze rule references in a Lark grammar.

    Parameters
    ----------
    grammar_text
        Contents of a ``.lark`` grammar file.
    known_terminals
        Terminal names provided from outside (e.g. imported from another
        grammar); references to them are not reported as missing.

    Returns
    -------
    GrammarReferenceAnalysis
        Undefined references and, if any, the shortest unguarded cycle.
    """

    parsed = _parse_grammar_text(grammar_text)
    rules = set(parsed.rule_defs)
    params = parsed.template_params

    # --- 1. undefined references (reported separately from cycles) ---------
    missing = {}
    bodies = list(parsed.rule_defs.items())
    if parsed.ignored:
        bodies.append(("%ignore", parsed.ignored))

    for owner, alternatives in bodies:
        for node in _iter_nodes(alternatives):
            if node.predicate:
                continue
            if node.kind is _NodeKind.RULE_REF:
                if node.name in rules or node.name in parsed.declared:
                    continue
                if node.name in params:
                    continue
                missing.setdefault(node.name, owner)
            elif node.kind is _NodeKind.TERMINAL_REF:
                if node.name in params:
                    continue
                if not _terminal_available(node.name, parsed, known_terminals):
                    missing.setdefault(node.name, owner)

    # --- 2. productive rules (least fixed point) ---------------------------
    productive = set()

    def node_can_terminate(node):
        if node.kind is _NodeKind.GROUP:
            if node.op in ("?", "*"):
                # Optional / star groups can derive the empty string, so
                # they never prevent the rest of the alternative from being
                # reached (e.g. "a: (a)* 'x'" derives "'x'").
                return True
            return any(
                all(child_can_terminate(c) for c in alt) for alt in node.children
            )
        if node.kind in (_NodeKind.STRING, _NodeKind.REGEXP, _NodeKind.RANGE):
            return True
        if node.filtered:
            return True
        if node.kind is _NodeKind.TERMINAL_REF:
            return True
        if node.kind is _NodeKind.RULE_REF and node.name in params:
            # Template parameter: replaced by a caller-provided symbol, which
            # is treated as a terminal guard for cycle analysis.
            return True
        return node.op == "?" or node.name in productive

    def child_can_terminate(node):
        # Items repeated zero or more times cannot block termination.
        if node.op in ("*", "?"):
            return True
        return node_can_terminate(node)

    def alt_can_terminate(alt):
        return all(node_can_terminate(node) for node in alt if not node.predicate)

    while True:
        added = {
            name
            for name, alts in parsed.rule_defs.items()
            if name not in productive and any(alt_can_terminate(alt) for alt in alts)
        }
        if not added:
            break
        productive |= added

    # --- 3. unguarded reference edges between non-productive rules ----------
    unguarded = {name: set() for name in rules}
    for name, alts in parsed.rule_defs.items():
        if name in productive:
            continue
        for alt in alts:
            _collect_unguarded_edges(alt, productive, name, unguarded, params)

    cycle = _shortest_cycle(unguarded)
    return GrammarReferenceAnalysis(
        rules=rules,
        missing=missing,
        cycle=cycle,
        productive=productive,
    )


def _node_terminates(node, productive, params=frozenset()):
    """Whether *node* itself must consume a terminal (non-nullable)."""

    if node.predicate or node.op in ("*", "?"):
        return False
    if node.kind is _NodeKind.GROUP:
        return any(
            all(_node_terminates(c, productive, params) for c in alt if not c.predicate)
            for alt in node.children
        )
    if node.kind in (_NodeKind.STRING, _NodeKind.REGEXP, _NodeKind.RANGE):
        return True
    if node.filtered:
        return True
    if node.kind is _NodeKind.TERMINAL_REF:
        return True
    if node.kind is _NodeKind.RULE_REF and node.name in params:
        return True
    return node.name in productive


def _collect_unguarded_edges(alt, productive, owner, edges, params=frozenset()):
    """Add edges for rule calls with no guaranteed terminal before them.

    Every item preceding the call must be guaranteed to consume a terminal,
    and the call itself must not be wrapped in an optional/star group.
    """

    for idx, node in enumerate(alt):
        if node.predicate:
            continue
        if idx > 0:
            # The immediately preceding item must itself consume a terminal
            # (transitively, via a non-nullable item). If it is only a rule
            # call, that call can loop forever before reaching this one.
            previous = alt[idx - 1]
            if previous.predicate or not _node_terminates(previous, productive, params):
                continue
        if node.kind is _NodeKind.RULE_REF and node.op not in ("?", "*"):
            if node.name not in params:
                edges[owner].add(node.name)
        elif node.kind is _NodeKind.GROUP:
            if node.op in ("?", "*"):
                # The whole group may derive the empty string; its internal
                # calls are guarded and do not create unguarded edges.
                continue
            for inner_alt in node.children:
                _collect_unguarded_edges(inner_alt, productive, owner, edges, params)


def _shortest_cycle(graph):
    """Find the shortest cycle in *graph* restricted to its own node set.

    Ties are broken by the lexicographically smallest path, and the smallest
    node of the cycle is reported first for stable, readable output.
    """

    best = None
    for start in sorted(graph):
        # Self-loop: the shortest possible cycle.
        if start in graph.get(start, ()):
            candidate = [start]
            if best is None or (1, candidate) < (len(best), best):
                best = candidate
            continue
        parents = {start: None}
        queue = [start]
        head = 0
        found_at = None
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt in sorted(graph.get(cur, ())):
                if nxt not in graph:
                    continue
                if nxt == start and cur != start:
                    found_at = cur
                    break
                if nxt not in parents:
                    parents[nxt] = cur
                    queue.append(nxt)
            if found_at is not None:
                break
        if found_at is None:
            continue
        path = []
        cur = found_at
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()
        k = min(range(len(path)), key=path.__getitem__)
        rotated = path[k:] + path[:k]
        if best is None or (len(rotated), rotated) < (len(best), best):
            best = rotated
    return best


def find_grammar_reference_cycle(grammar_text):
    """Return the shortest unguarded rule-reference cycle, or ``None``.

    Undefined references do not affect cycle detection; they are diagnosed
    separately by :func:`analyze_grammar_references`.
    """

    return analyze_grammar_references(grammar_text).cycle


def preprocess_grammar(grammar_text, known_terminals=None):
    """Validate and preprocess Lark grammar text.

    Diagnostics run in a fixed order: undefined references first
    (:class:`GrammarReferenceNotFoundError`), then unguarded reference cycles
    (:class:`GrammarReferenceCycleError`). The grammar text is returned
    unchanged, so this can be used as a validating wrapper before handing the
    grammar to a parser or backend.
    """

    analysis = analyze_grammar_references(grammar_text, known_terminals)
    if analysis.missing:
        raise GrammarReferenceNotFoundError(analysis.missing)
    if analysis.cycle:
        raise GrammarReferenceCycleError(analysis.cycle)
    return grammar_text


def read_grammar(
    grammar_file_name,
    base_grammar_path=GRAMMAR_PATH,
    validate=True,
):
    """Read grammar file from default grammar path.

    Parameters
    ----------
    grammar_file_name
        The name of the grammar file to read.
    base_grammar_path
        The path to the directory containing the grammar file.
    validate
        When ``True`` (the default) the grammar is preprocessed first, so
        undefined references and unguarded reference cycles raise
        :class:`GrammarPreprocessingError` subclasses.

    Returns
    -------
    str
        The contents of the grammar file.

    """

    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        contents = file.read()
    if validate:
        preprocess_grammar(contents)
    return contents


arithmetic = read_grammar("arithmetic.lark")
json = read_grammar("json.lark")
