from sqlalchemy.dialects import postgresql

from src.memory.service import recall_statement


def test_recall_sql_filters_by_distance_and_project_and_orders_by_distance():
    stmt = recall_statement("proj-1", [0.1] * 768, k=3, max_distance=0.45)
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": False}))
    assert "<=>" in sql                                  # pgvector cosine-distance operator
    assert "memory_chunks.project_id" in sql
    assert sql.count("<=>") >= 2                         # filter + ORDER BY (index-friendly)
    assert "ORDER BY" in sql and "LIMIT" in sql
    params = stmt.compile(dialect=postgresql.dialect()).params
    assert 0.45 in params.values() and 3 in params.values()


def _sql(persona):
    stmt = recall_statement("proj-1", [0.1] * 768, k=3, max_distance=0.45, persona=persona)
    comp = stmt.compile(dialect=postgresql.dialect())
    return str(comp), comp.params


def test_nonstop_recalls_every_partition_without_a_source_filter():
    for persona in ("nonstop", None):
        sql, params = _sql(persona)
        assert "memory_chunks.source" not in sql
        assert not any(isinstance(v, str) and v.startswith("message") for v in params.values())


def test_specialist_recalls_only_its_own_exchanges_and_shared_notes():
    sql, params = _sql("coder")
    assert "memory_chunks.source" in sql and " OR " in sql and "NOT" in sql
    assert "message:coder" in params.values()          # its own partition
    assert "message" in params.values()                # ... and "not a message" = shared notes
    assert "message:security" not in params.values()   # never another mode's


def test_each_specialist_gets_its_own_partition_tag():
    from src.memory.service import message_source, sees_everything
    tags = {p: message_source(p) for p in ("executive", "coder", "architect", "security", "researcher", "writer")}
    assert len(set(tags.values())) == 6 and all(t.startswith("message:") for t in tags.values())
    assert message_source(None) == "message"
    assert sees_everything("nonstop") and sees_everything(None) and not sees_everything("coder")
