"""storage v2：session_id 贯通（第10期）。"""

from __future__ import annotations

import json

from app.agent.storage import SESSION_VERSION, load_session, save_session


def test_save_generates_session_id(tmp_path):
    path = str(tmp_path / "s1.json")
    save_session(path, [{"role": "user", "content": "hi"}], "sum")
    data = json.loads(tmp_path.joinpath("s1.json").read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["session_id"]
    assert len(data["session_id"]) == 32  # uuid4().hex


def test_save_explicit_session_id_roundtrip(tmp_path):
    path = str(tmp_path / "s2.json")
    save_session(path, [{"role": "user", "content": "hi"}], None,
                 session_id="abc123")
    loaded = load_session(path)
    assert loaded["session_id"] == "abc123"
    assert loaded["messages"] == [{"role": "user", "content": "hi"}]
    assert loaded["summary"] is None


def test_load_returns_session_id_and_stm(tmp_path):
    path = str(tmp_path / "s3.json")
    save_session(path, [{"role": "user", "content": "hi"}], "sum",
                 short_term_memory={"k": "v"}, session_id="sid-1")
    loaded = load_session(path)
    assert loaded["session_id"] == "sid-1"
    assert loaded["short_term_memory"] == {"k": "v"}


def test_v1_file_without_session_id(tmp_path):
    """v1 文件没有 session_id 键 → load 返回 None（调用方生成新 ID）。"""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "version": 1,
        "summary": "s",
        "messages": [{"role": "user", "content": "hi"}],
        "short_term_memory": None,
    }, ensure_ascii=False), encoding="utf-8")
    loaded = load_session(str(path))
    assert loaded["session_id"] is None
    assert loaded["messages"]


def test_load_missing_or_damaged(tmp_path):
    assert load_session(str(tmp_path / "none.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_session(str(bad)) is None


def test_atomic_write_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "s4.json")
    save_session(path, [{"role": "user", "content": "hi"}], None)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []