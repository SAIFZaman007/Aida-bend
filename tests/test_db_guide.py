import pytest

from src.core.database import _raise_pgvector_guide


def test_guide_includes_original_error_and_is_not_windows_only():
    with pytest.raises(RuntimeError) as ei:
        _raise_pgvector_guide(Exception('extension "vector" is not available'))
    msg = str(ei.value)
    assert 'extension "vector" is not available' in msg   # real cause is shown
    assert "pgvector/pgvector" in msg                     # Docker/Coolify route
    assert "permission denied" in msg                     # other cause covered
    assert "WINDOWS_SETUP" not in msg                     # no reference to a missing file
