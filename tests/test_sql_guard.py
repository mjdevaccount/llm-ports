"""sql_guard -- the read-only cage around agent-written SQL. Fail-closed is the contract: if any
rejection here turns green, the guard admits a write, a foreign schema or a dangerous call."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
from sqlglot import expressions as exp

from llm_ports.sql_guard import FORBIDDEN_NODES, POSTGRES_DENY_FUNCTIONS, SqlGuardError, SqlGuardPolicy, guard

POLICY = SqlGuardPolicy(allowed_schemas=("app", "reporting"), deny_functions=POSTGRES_DENY_FUNCTIONS, max_rows=200)


def g(sql: str, policy: SqlGuardPolicy = POLICY) -> str:
    return guard(sql, policy)


@pytest.mark.parametrize(("sql", "want"), [
    ("SELECT id, ts FROM app.events", "SELECT id, ts FROM app.events LIMIT 200"),
    ("SELECT * FROM app.events LIMIT 5", "SELECT * FROM app.events LIMIT 5"),
    ("SELECT * FROM app.events LIMIT 100000", "SELECT * FROM app.events LIMIT 200"),
    ("SELECT id /* sneaky */ FROM app.events -- trailing", "SELECT id FROM app.events LIMIT 200"),
    ("SELECT id FROM app.events UNION ALL SELECT id FROM reporting.daily",
     "SELECT id FROM app.events UNION ALL SELECT id FROM reporting.daily LIMIT 200"),
    ("WITH d AS (SELECT id FROM app.events) SELECT d.id FROM d JOIN reporting.daily r ON r.id = d.id",
     "WITH d AS (SELECT id FROM app.events) SELECT d.id FROM d JOIN reporting.daily AS r ON r.id = d.id LIMIT 200"),
])
def test_admitted_shapes_are_regenerated_with_the_limit_enforced(sql: str, want: str) -> None:
    assert g(sql) == want


@pytest.mark.parametrize("bad", [
    "INSERT INTO app.events (id) VALUES (1)", "UPDATE app.events SET id = 9", "DELETE FROM app.events",
    "DROP TABLE app.events", "TRUNCATE app.events", "CREATE TABLE app.evil (x int)",
    "ALTER TABLE app.events ADD COLUMN x int", "GRANT ALL ON app.events TO public", "SET ROLE admin",
    "SELECT * FROM app.events FOR UPDATE", "SELECT * FROM app.events FOR SHARE",
    "SELECT * INTO app.evil FROM app.events", "EXPLAIN ANALYZE SELECT 1", "   ",
    "WITH d AS (DELETE FROM app.events RETURNING *) SELECT * FROM d",
    "DO $$ BEGIN NULL; END $$", "COPY app.events TO '/tmp/exfil.csv'", "VACUUM app.events",
    "SELECT 1; -- comment\nDROP TABLE app.events", "SELECT 1; SELECT 2",
])
def test_writes_utilities_locks_and_stacks_are_refused(bad: str) -> None:
    with pytest.raises(SqlGuardError):
        g(bad)


@pytest.mark.parametrize("sql", ["SELECT * FROM pg_catalog.pg_roles", "SELECT * FROM information_schema.tables",
                                 "SELECT * FROM ops.audit", "SELECT * FROM public.anything"])
def test_a_schema_outside_the_policy_is_not_queryable(sql: str) -> None:
    with pytest.raises(SqlGuardError, match=r"not queryable \(allowed: app, reporting\)"):
        g(sql)


def test_an_unqualified_table_is_refused_and_a_cte_name_is_not_a_table() -> None:
    with pytest.raises(SqlGuardError, match=r"schema-qualified \(app\.\* or reporting\.\*\)"):
        g("SELECT * FROM events")
    assert "RECURSIVE" in g("WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 9) "
                            "SELECT max(n) FROM r")


@pytest.mark.parametrize("sql", ["SELECT pg_sleep(60)", "SELECT pg_read_file('/etc/passwd')",
                                 "SELECT id FROM app.events WHERE pg_sleep(9) IS NULL",
                                 "SELECT (SELECT set_config('statement_timeout', '0', true)) FROM app.events"])
def test_a_denied_function_anywhere_in_the_tree_is_refused(sql: str) -> None:
    with pytest.raises(SqlGuardError, match="not permitted"):
        g(sql)


def test_a_non_integer_limit_is_refused() -> None:
    with pytest.raises(SqlGuardError, match="plain integer"):
        g("SELECT * FROM app.events LIMIT ALL")


def test_every_policy_field_is_read() -> None:
    sql = "SELECT avg(x) FROM reporting.daily"
    assert "AVG(x)" in g(sql)
    with pytest.raises(SqlGuardError, match="not queryable"):
        g(sql, replace(POLICY, allowed_schemas=("app",)))
    with pytest.raises(SqlGuardError, match="'avg' is not permitted"):
        g(sql, replace(POLICY, deny_functions=POLICY.deny_functions | {"avg"}))
    assert g(sql, replace(POLICY, max_rows=50)).endswith("LIMIT 50")
    with pytest.raises(SqlGuardError, match="forbidden construct: UNION"):
        g("SELECT 1 UNION SELECT 2", replace(POLICY, forbidden_nodes=(*FORBIDDEN_NODES, exp.Union)))


def test_a_policy_may_tighten_but_never_loosen() -> None:
    # MUST FIRE: drop the `missing` check and a policy without Delete builds.
    with pytest.raises(dataclasses.FrozenInstanceError):
        POLICY.max_rows = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="single_statement"):
        replace(POLICY, single_statement=False)
    with pytest.raises(ValueError, match=r"forbidden_nodes may only grow.*Delete"):
        replace(POLICY, forbidden_nodes=tuple(n for n in FORBIDDEN_NODES if n is not exp.Delete))
    with pytest.raises(ValueError, match="max_rows"):
        replace(POLICY, max_rows=0)
    with pytest.raises(ValueError, match="lower-case"):
        replace(POLICY, allowed_schemas=("App",))
    with pytest.raises(ValueError, match="lower-case"):
        replace(POLICY, deny_functions=frozenset({"PG_SLEEP"}))
