"""A few common Lark grammars."""

from collections import deque
from pathlib import Path

from lark.grammar import NonTerminal, Terminal
from lark.load_grammar import GrammarBuilder
from lark.tree import Tree

from outlines.exceptions import OutlinesError

GRAMMAR_PATH = Path(__file__).parent / "grammars"


class GrammarReferenceError(OutlinesError):
    """Raised when a grammar has invalid references or non-productive cycles.

    Two distinct problems are reported:

    * **Missing references** -- a rule references a rule or terminal that is
      not defined anywhere in the grammar (including ``%import``\\ s).
    * **Non-productive reference cycles** -- a cycle in the rule-reference
      graph with no terminal exit.  Recursive rules that *do* have a base case
      (e.g. ``start: "a" start | "b"``) are productive and therefore legal.
    """


def read_grammar(
    grammar_file_name: str,
    base_grammar_path: Path = GRAMMAR_PATH,
    validate: bool = True,
) -> str:
    """Read grammar file from default grammar path.

    Parameters
    ----------
    grammar_file_name
        The name of the grammar file to read.
    base_grammar_path
        The path to the directory containing the grammar file.
    validate
        If ``True`` (default), run :func:`preprocess_grammar` on the contents
        to diagnose missing references and non-productive reference cycles
        before returning.

    Returns
    -------
    str
        The contents of the grammar file.

    Raises
    ------
    GrammarReferenceError
        If the grammar references undefined rules/terminals or contains a
        non-productive reference cycle.

    """
    full_path = base_grammar_path / grammar_file_name
    with open(full_path) as file:
        grammar = file.read()
    if validate:
        preprocess_grammar(grammar, source=grammar_file_name)
    return grammar


# ---------------------------------------------------------------------------
# Grammar preprocessing: reference-cycle localization
# ---------------------------------------------------------------------------

def preprocess_grammar(grammar: str, source: str = "<string>") -> str:
    """Validate a Lark grammar for missing references and non-productive cycles.

    This is the grammar preprocessing entry point.  It parses the grammar
    (without triggering Lark's own validation, so that missing references can
    be collected rather than aborting on the first one), then:

    1. Collects every rule and terminal definition.
    2. Finds **missing references** -- rules that reference a rule or terminal
       name which is not defined.
    3. Computes the set of **productive rules** (rules that can derive a
       string of terminals) via fixed-point iteration.
    4. Finds the **shortest non-productive reference cycle** -- a cycle in the
       subgraph of rules that can never reach a terminal.  Legal recursion
       (recursive rules with a productive base case) is *not* reported.

    If any problem is found, a :class:`GrammarReferenceError` is raised whose
    message lists the missing references and the shortest readable cycle
    path.

    Parameters
    ----------
    grammar
        The Lark grammar string to validate.
    source
        A label for the grammar source, used in error messages (e.g. the file
        name).

    Returns
    -------
    str
        The unchanged grammar string if it is valid.

    Raises
    ------
    GrammarReferenceError
        If the grammar has missing references or a non-productive cycle.

    """
    rules, terminals = _parse_grammar(grammar)

    missing = _find_missing_references(rules, terminals)
    productive = _find_productive_rules(rules)
    cycle = _find_shortest_cycle(rules, productive)

    if missing or cycle:
        raise GrammarReferenceError(_format_diagnostics(source, missing, cycle))

    return grammar


def _parse_grammar(grammar: str) -> tuple[dict[str, Tree], set[str]]:
    """Parse a Lark grammar into rule definitions and terminal names.

    Uses :class:`GrammarBuilder` directly (without ``build()``\\ ) so that
    missing references do not abort parsing.  The builder populates
    ``_definitions`` with every rule and terminal, including ``%import``\\ ed
    terminals.

    Returns
    -------
    rules
        Mapping of rule name -> the rule's expansion AST (a
        ``Tree('expansions', ...)``).
    terminals
        Set of terminal names defined in the grammar.

    """
    builder = GrammarBuilder(global_keep_all_tokens=False, import_paths=[])
    builder.load_grammar(grammar, "<string>")

    rules: dict[str, Tree] = {}
    terminals: set[str] = set()
    for name, definition in builder._definitions.items():
        if definition.is_term:
            terminals.add(str(name))
        else:
            rules[str(name)] = definition.tree

    return rules, terminals


def _collect_references(tree: Tree) -> tuple[set[str], set[str]]:
    """Collect the rule and terminal names referenced by an AST.

    Returns
    -------
    rule_refs
        Names referenced via ``NonTerminal`` (rule references).
    term_refs
        Names referenced via ``Terminal`` (terminal references).

    """
    rule_refs: set[str] = set()
    term_refs: set[str] = set()

    def walk(node, skip_alias: bool = False) -> None:
        if isinstance(node, NonTerminal):
            if not skip_alias:
                rule_refs.add(node.name)
        elif isinstance(node, Terminal):
            term_refs.add(node.name)
        elif isinstance(node, Tree):
            if node.data == "alias":
                # Tree('alias', [expansion, alias_name]) -- the second child
                # is the alias name, not a rule reference.
                if node.children:
                    walk(node.children[0])
                return
            for child in node.children:
                walk(child)

    walk(tree)
    return rule_refs, term_refs


def _find_missing_references(
    rules: dict[str, Tree], terminals: set[str]
) -> list[tuple[str, str, str]]:
    """Find references to rules or terminals that are not defined.

    Returns a list of ``(referencing_rule, kind, missing_name)`` tuples,
    where ``kind`` is ``"rule"`` or ``"terminal"``.

    """
    missing: list[tuple[str, str, str]] = []
    defined_rules = set(rules)
    for rule_name, tree in rules.items():
        rule_refs, term_refs = _collect_references(tree)
        for ref in sorted(rule_refs):
            if ref not in defined_rules:
                missing.append((rule_name, "rule", ref))
        for ref in sorted(term_refs):
            if ref not in terminals:
                missing.append((rule_name, "terminal", ref))
    return missing


def _is_productive(node, productive: set[str]) -> bool:
    """Return ``True`` if the AST node can derive a string of terminals.

    This is a structural productivity check: a node is productive if it has
    at least one derivation that reaches terminals without getting stuck in
    a non-productive cycle.  Terminals (literals, regexes, terminal
    references) are always productive; nonterminals are productive iff their
    rule is in the productive set.

    """
    if isinstance(node, NonTerminal):
        return node.name in productive
    if isinstance(node, Terminal):
        return True
    if not isinstance(node, Tree):
        # Token (literal / regex) or other leaf -- always productive.
        return True

    if node.data == "expansions":
        # One alternative being productive is enough.
        return any(_is_productive(child, productive) for child in node.children)
    if node.data == "expansion":
        # Every element in the sequence must be productive.
        return all(_is_productive(child, productive) for child in node.children)
    if node.data == "value":
        return all(_is_productive(child, productive) for child in node.children)
    if node.data == "maybe":
        # Optional ([ ... ]) can derive the empty string.
        return True
    if node.data == "expr":
        # Postfix operator: '?' and '*' can derive the empty string;
        # '+' needs the repeated element to be productive.
        if len(node.children) >= 2:
            op = node.children[-1]
            if getattr(op, "value", None) in ("?", "*"):
                return True
        return all(_is_productive(child, productive) for child in node.children[:-1])
    if node.data == "literal":
        return True
    if node.data == "alias":
        # Tree('alias', [expansion, alias_name]) -- productivity is determined
        # by the expansion (first child), not the alias name.
        if node.children:
            return _is_productive(node.children[0], productive)
        return False
    if node.data == "template_usage":
        # First child is the referenced rule name.
        if node.children and isinstance(node.children[0], NonTerminal):
            return node.children[0].name in productive
        return False

    # Unknown construct: be conservative and treat as productive so that we
    # don't false-positive on grammars using advanced Lark features.
    return True


def _find_productive_rules(rules: dict[str, Tree]) -> set[str]:
    """Compute the set of productive rules via fixed-point iteration.

    A rule is productive if at least one of its alternatives can derive a
    string of terminals.  The fixed-point starts with no productive rules
    and repeatedly adds rules that have a productive alternative, until no
    more can be added.

    """
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, tree in rules.items():
            if name in productive:
                continue
            # tree is Tree('expansions', [Tree('expansion', ...), ...])
            if any(
                _is_productive(alt, productive) for alt in tree.children
            ):
                productive.add(name)
                changed = True
    return productive


def _find_shortest_cycle(
    rules: dict[str, Tree], productive: set[str]
) -> list[str] | None:
    """Find the shortest non-productive reference cycle in the grammar.

    Only rules that are *not* productive are considered, because a cycle
    through a productive rule has a terminal exit and is therefore legal
    recursion.

    Returns the cycle as a list of rule names ``[v0, v1, ..., vk]`` where
    ``vk == v0``, or ``None`` if no non-productive cycle exists.

    """
    # Build the reference graph restricted to non-productive rules.
    graph: dict[str, set[str]] = {}
    for name, tree in rules.items():
        if name in productive:
            continue
        rule_refs, _ = _collect_references(tree)
        graph[name] = {
            ref for ref in rule_refs if ref in rules and ref not in productive
        }

    best: list[str] | None = None
    for start in graph:
        cycle = _bfs_shortest_cycle(start, graph)
        if cycle is not None and (best is None or len(cycle) < len(best)):
            best = cycle
            if len(best) == 1:
                # Self-loop is the shortest possible cycle.
                break
    return best


def _bfs_shortest_cycle(
    start: str, graph: dict[str, set[str]]
) -> list[str] | None:
    """BFS from ``start`` to find the shortest cycle passing through it.

    Returns the cycle as ``[start, ..., start]``, or ``None`` if ``start``
    is not part of any cycle.

    """
    parent: dict[str, str | None] = {start: None}
    queue: deque[str] = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor == start:
                # Reconstruct the path start -> ... -> node, then close it.
                path: list[str] = []
                cur: str | None = node
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                path.append(start)
                return path
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return None


def _format_diagnostics(
    source: str,
    missing: list[tuple[str, str, str]],
    cycle: list[str] | None,
) -> str:
    """Format the grammar diagnostics into a readable error message."""
    lines = [f"Grammar reference error in {source!r}:"]

    if missing:
        lines.append("  Missing references:")
        for referencing_rule, kind, name in missing:
            lines.append(
                f"    - rule {referencing_rule!r} references undefined "
                f"{kind} {name!r}"
            )

    if cycle:
        lines.append("  Non-productive reference cycle (shortest):")
        lines.append("    " + " -> ".join(cycle))

    return "\n".join(lines)


arithmetic = read_grammar("arithmetic.lark")
json = read_grammar("json.lark")
