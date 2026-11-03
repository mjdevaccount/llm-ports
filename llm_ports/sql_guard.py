"""sql_guard -- a read-only AST guard for SQL an agent wrote (extra `[sql]`, sqlglot).

    POLICY = SqlGuardPolicy(allowed_schemas=("reporting",), deny_functions=POSTGRES_DENY_FUNCTIONS,
                            max_rows=200)
    safe_sql = guard(agent_sql, POLICY)      # raises SqlGuardError with a reason the agent can act on

One layer of a defense stack, not the stack: run the result as a role that cannot write, in a
READ ONLY transaction with a statement timeout. What this layer does: parse with sqlglot, validate
the tree, and REGENERATE the SQL from the tree -- the returned string is the only thing to execute,
so comments and formatting tricks do not survive the parser.

The rules (a frozen `SqlGuardPolicy` holds the data; the consumer owns its schema allowlist):
- exactly one statement, and it is a SELECT (UNION / CTE / subquery fine);
- no DML / DDL / transaction / utility node anywhere in the tree (`FORBIDDEN_NODES`), no
  `SELECT ... INTO`;
- every real table schema-qualified and inside `allowed_schemas` (CTE names exempt, so the system
  catalogs are denied unless a policy names them);
- no function in `deny_functions` (`POSTGRES_DENY_FUNCTIONS`: sleep, file, backend-signal,
  advisory-lock, dblink and xml-dump functions);
- LIMIT forced to <= `max_rows` (injected when absent; a non-integer LIMIT is refused).

A policy may TIGHTEN the structural rules but never loosen them: `forbidden_nodes` must keep every
type in `FORBIDDEN_NODES`, and `single_statement` cannot be turned off. Both are refused at
construction, so no instance exists that widens what the guard admits. Fail-closed: anything
unparseable raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import expressions as exp

#: Node types that never appear anywhere in an admitted tree. `exp.Command` is sqlglot's catch-all
#: for a raw or unmodelled statement -- fail-closed includes it.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
    exp.Alter, exp.Grant, exp.TruncateTable, exp.Set, exp.Command,
    exp.Transaction, exp.Commit, exp.Rollback, exp.Use, exp.Copy, exp.Lock,
)

#: PostgreSQL functions an agent's read has no business calling (lower-case, as policies hold them).
POSTGRES_DENY_FUNCTIONS: frozenset[str] = frozenset({
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "pg_rotate_logfile", "lo_import", "lo_export", "set_config",
    "dblink", "dblink_connect", "dblink_exec",
    "pg_advisory_lock", "pg_advisory_xact_lock", "pg_try_advisory_lock",
    "query_to_xml", "database_to_xml", "table_to_xml",
})


class SqlGuardError(ValueError):
    """The statement was refused; the message is a reason the agent can act on."""


@dataclass(frozen=True)
class SqlGuardPolicy:
    """What the guard admits. `allowed_schemas` is an ordered tuple so the error text is stable."""

    allowed_schemas: tuple[str, ...]
    deny_functions: frozenset[str]
    max_rows: int
    forbidden_nodes: tuple[type[exp.Expression], ...] = FORBIDDEN_NODES
    single_statement: bool = True
    dialect: str = "postgres"

    def __post_init__(self) -> None:
        if not self.single_statement:
            raise ValueError("SqlGuardPolicy.single_statement=False is not a supported policy")
        missing = [n.__name__ for n in FORBIDDEN_NODES if n not in self.forbidden_nodes]
        if missing:
            raise ValueError(f"SqlGuardPolicy.forbidden_nodes may only grow; missing {missing}")
        if self.max_rows < 1:
            raise ValueError("SqlGuardPolicy.max_rows must be >= 1")
        if any(s != s.lower() for s in self.allowed_schemas):
            raise ValueError("SqlGuardPolicy.allowed_schemas must be lower-case")
        if any(f != f.lower() for f in self.deny_functions):
            raise ValueError("SqlGuardPolicy.deny_functions must be lower-case")

    @property
    def schemas_text(self) -> str:
        return ", ".join(self.allowed_schemas)


def _check_tables(tree: exp.Expression, policy: SqlGuardPolicy) -> None:
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = (t.name or "").lower()
        schema = (t.db or "").lower()
        if not schema:
            if name in ctes:
                continue  # a CTE reference, not a real table
            qualified = " or ".join(f"{s}.*" for s in policy.allowed_schemas)
            raise SqlGuardError(f"table '{name}' must be schema-qualified ({qualified})")
        if schema not in policy.allowed_schemas:
            raise SqlGuardError(f"schema '{schema}' is not queryable (allowed: {policy.schemas_text})")


def _check_functions(tree: exp.Expression, policy: SqlGuardPolicy) -> None:
    for fn in tree.find_all(exp.Func):
        name = (fn.name or fn.sql_name() or "").lower()
        if name in policy.deny_functions:
            raise SqlGuardError(f"function '{name}' is not permitted")
    for anon in tree.find_all(exp.Anonymous):
        if (anon.name or "").lower() in policy.deny_functions:
            raise SqlGuardError(f"function '{anon.name}' is not permitted")


def _enforce_limit(tree: exp.Expression, policy: SqlGuardPolicy) -> exp.Expression:
    limit = tree.args.get("limit") if isinstance(tree, (exp.Select, exp.Union)) else None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (TypeError, ValueError) as e:
            raise SqlGuardError("LIMIT must be a plain integer") from e
        if current <= policy.max_rows:
            return tree
    return tree.limit(policy.max_rows)


def guard(sql: str, policy: SqlGuardPolicy) -> str:
    """Validate `sql` under `policy` and return the regenerated statement to execute (comments
    stripped). Raises SqlGuardError."""
    if not sql or not sql.strip():
        raise SqlGuardError("empty query")
    try:
        statements = sqlglot.parse(sql, read=policy.dialect)
    except Exception as e:
        raise SqlGuardError(f"could not parse SQL: {e}") from e
    statements = [s for s in statements if s is not None]
    # `single_statement` is True by construction, so this check is unconditional.
    if len(statements) != 1:
        raise SqlGuardError(f"exactly one statement required (got {len(statements)})")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SqlGuardError("only SELECT statements are permitted")
    for node_type in policy.forbidden_nodes:
        if tree.find(node_type) is not None:
            raise SqlGuardError(f"forbidden construct: {node_type.__name__.upper()}")
    if tree.find(exp.Into) is not None:
        raise SqlGuardError("an INTO clause (SELECT ... INTO) is not permitted")
    _check_tables(tree, policy)
    _check_functions(tree, policy)
    tree = _enforce_limit(tree, policy)
    return tree.sql(dialect=policy.dialect, comments=False)


__all__ = ["FORBIDDEN_NODES", "POSTGRES_DENY_FUNCTIONS", "SqlGuardError", "SqlGuardPolicy", "guard"]
