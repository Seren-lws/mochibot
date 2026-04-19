"""Tests for workspace skill — diary read/write and file editing."""

import os
import tempfile

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_temp_db.close()
os.environ["MOCHIBOT_DB_PATH"] = _temp_db.name

import mochi.diary as diary_mod
from mochi.diary import DailyFile
from mochi.skills.base import SkillContext, SkillResult

UTC = timezone.utc


@pytest.fixture
def test_diary(tmp_path, monkeypatch):
    monkeypatch.setattr(diary_mod, "TZ", UTC)
    monkeypatch.setattr(diary_mod, "_diary_date",
                        lambda: datetime(2025, 6, 15, 10, 0, tzinfo=UTC))
    monkeypatch.setattr(diary_mod, "_today_str", lambda: "2025-06-15")
    monkeypatch.setattr(diary_mod, "_now_time", lambda: "10:00")

    d = DailyFile(
        path=tmp_path / "diary.md",
        label="Diary",
        max_lines=50,
        sections=("今日状態", "今日日記"),
        section_max_lines={"今日状態": 20, "今日日記": 30},
    )
    return d


@pytest.fixture
def workspace(test_diary, tmp_path, monkeypatch):
    from mochi.skills.workspace.handler import WorkspaceSkill
    import mochi.skills.workspace.handler as ws_mod
    monkeypatch.setattr(ws_mod, "diary", test_diary)
    monkeypatch.setattr(ws_mod, "_DATA_DIR", tmp_path)
    skill = WorkspaceSkill()
    skill._name = "workspace"
    return skill


def _ctx(tool_name: str, args: dict) -> SkillContext:
    return SkillContext(
        trigger="tool_call",
        user_id=1,
        tool_name=tool_name,
        args=args,
    )


class TestWriteDiary:

    @pytest.mark.asyncio
    async def test_basic_write(self, workspace, test_diary):
        result = await workspace.execute(_ctx("write_diary", {"entry": "went to gym"}))
        assert "Recorded" in result.output
        content = test_diary.read(section="今日日記")
        assert "went to gym" in content

    @pytest.mark.asyncio
    async def test_source_is_chat(self, workspace, test_diary):
        await workspace.execute(_ctx("write_diary", {"entry": "test entry"}))
        raw = test_diary.read_raw()
        assert "[10:00]" in raw
        assert "💭" not in raw

    @pytest.mark.asyncio
    async def test_empty_entry_rejected(self, workspace):
        result = await workspace.execute(_ctx("write_diary", {"entry": ""}))
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_no_entry_rejected(self, workspace):
        result = await workspace.execute(_ctx("write_diary", {}))
        assert "Error" in result.output


class TestReadDiary:

    @pytest.mark.asyncio
    async def test_read_today(self, workspace, test_diary):
        test_diary.append("morning run", source="chat", section="今日日記")
        result = await workspace.execute(_ctx("read_diary", {}))
        assert "morning run" in result.output

    @pytest.mark.asyncio
    async def test_read_empty(self, workspace):
        result = await workspace.execute(_ctx("read_diary", {}))
        assert "empty" in result.output.lower()

    @pytest.mark.asyncio
    async def test_read_archive(self, workspace, test_diary):
        archive_dir = test_diary.path.parent / "diary_archive"
        archive_dir.mkdir()
        (archive_dir / "2025-05.md").write_text(
            "# Diary 2025-05-10 Saturday\n"
            "- [09:00] had coffee\n\n"
            "# Diary 2025-05-11 Sunday\n"
            "- [10:00] went hiking\n",
            encoding="utf-8",
        )
        result = await workspace.execute(_ctx("read_diary", {"date": "2025-05-10"}))
        assert "had coffee" in result.output
        assert "went hiking" not in result.output

    @pytest.mark.asyncio
    async def test_read_archive_not_found(self, workspace):
        result = await workspace.execute(_ctx("read_diary", {"date": "2020-01-01"}))
        assert "No diary archive" in result.output


class TestEditFile:

    @pytest.mark.asyncio
    async def test_read_file(self, workspace, tmp_path):
        (tmp_path / "test.md").write_text("hello world", encoding="utf-8")
        result = await workspace.execute(_ctx("edit_file", {
            "action": "read", "path": "test.md",
        }))
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, workspace, tmp_path):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "write", "path": "test.md", "content": "new content",
        }))
        assert "OK" in result.output
        assert (tmp_path / "test.md").read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, workspace):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "read", "path": "nope.md",
        }))
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_reject_non_md(self, workspace):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "read", "path": "secret.env",
        }))
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_reject_path_traversal(self, workspace):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "read", "path": "../../etc/passwd",
        }))
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_write_no_content(self, workspace):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "write", "path": "test.md",
        }))
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_unknown_action(self, workspace):
        result = await workspace.execute(_ctx("edit_file", {
            "action": "delete", "path": "test.md",
        }))
        assert "Error" in result.output
