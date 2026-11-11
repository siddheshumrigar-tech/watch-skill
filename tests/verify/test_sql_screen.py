"""The `sqlite_query` write screen reads SQL tokens, not substrings.

It looked for a write keyword anywhere in the text, so a column called
`replace`, the function `replace()`, or the string literal 'drop table x' was
refused, while the mode=ro handle was what actually kept the database safe.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from watch_skill.verify import Check, CheckContext, CheckStatus
from watch_skill.verify.checks import run_check


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "app.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE t (id INTEGER, name TEXT, "replace" TEXT)')
    conn.execute("INSERT INTO t VALUES (1, 'banana', 'x')")
    conn.commit()
    conn.close()
    return path


def _run(tmp_path: Path, sql: str, **params):
    return run_check(
        Check(id="s", type="sqlite_query", params={"database": "app.db", "sql": sql, **params}),
        CheckContext(working_dir=str(tmp_path), allowed_roots=[str(tmp_path)]),
    )


def test_a_column_named_replace_is_readable(tmp_path: Path, db: Path) -> None:
    result = _run(tmp_path, "SELECT replace FROM t", equals=[{"replace": "x"}])
    assert result.status is CheckStatus.PASS, result.summary


def test_a_keyword_inside_a_string_literal_is_data(tmp_path: Path, db: Path) -> None:
    result = _run(tmp_path, "SELECT 'drop table x' AS s", equals=[{"s": "drop table x"}])
    assert result.status is CheckStatus.PASS, result.summary


def test_the_replace_function_is_allowed(tmp_path: Path, db: Path) -> None:
    result = _run(tmp_path, "SELECT replace(name,'a','b') AS n FROM t",
                  equals=[{"n": "bbnbnb"}])
    assert result.status is CheckStatus.PASS, result.summary


def test_comments_and_quoted_identifiers_are_not_statements(tmp_path: Path, db: Path) -> None:
    sql = 'SELECT "replace" -- delete everything\nFROM t /* drop */'
    assert _run(tmp_path, sql, row_count=1).status is CheckStatus.PASS


def test_a_cte_that_selects_is_allowed(tmp_path: Path, db: Path) -> None:
    sql = "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM x WHERE n < 3) SELECT n FROM x"
    assert _run(tmp_path, sql, row_count=3).status is CheckStatus.PASS


def test_values_is_allowed(tmp_path: Path, db: Path) -> None:
    assert _run(tmp_path, "VALUES (1), (2)", row_count=2).status is CheckStatus.PASS


@pytest.mark.parametrize("sql", [
    "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x",
    "WITH x AS MATERIALIZED (SELECT 1), y AS (SELECT 2) DELETE FROM t",
    "INSERT INTO t VALUES (2, 'x', 'y')",
    "select 1; drop table t",
    "REPLACE INTO t VALUES (1, 'a', 'b')",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE 'other.db' AS o",
    "",
])
def test_writes_and_multiple_statements_are_refused(tmp_path: Path, db: Path, sql: str) -> None:
    result = _run(tmp_path, sql)
    assert result.status is CheckStatus.ERROR, sql
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    conn.close()


def test_a_semicolon_inside_a_literal_is_one_statement(tmp_path: Path, db: Path) -> None:
    result = _run(tmp_path, "SELECT 'a;b' AS s", equals=[{"s": "a;b"}])
    assert result.status is CheckStatus.PASS, result.summary


def test_tokens_strip_comments_literals_and_identifiers() -> None:
    from watch_skill.verify.checks import _sql_tokens, _statement_verb

    tokens = _sql_tokens("SELECT 'it''s' , [drop] , `x` -- y\n FROM t /* z */ ;")
    assert tokens == ["select", ",", ",", "from", "t", ";"]
    assert _statement_verb(_sql_tokens("WITH a(x) AS NOT MATERIALIZED (SELECT 1) SELECT x FROM a")) == "select"
    assert _statement_verb(_sql_tokens("with a as (select 1) update t set id = 1")) == "update"
