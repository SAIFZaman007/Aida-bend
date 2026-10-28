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
