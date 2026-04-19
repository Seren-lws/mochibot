"""Workspace skill — diary read/write and markdown file editing."""

import logging
from pathlib import Path

from mochi.diary import diary
from mochi.skills.base import Skill, SkillContext, SkillResult

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


class WorkspaceSkill(Skill):

    async def execute(self, context: SkillContext) -> SkillResult:
        tool_name, args = context.tool_name, context.args
        if tool_name == "write_diary":
            return SkillResult(output=self._write_diary(args))
        elif tool_name == "read_diary":
            return SkillResult(output=self._read_diary(args))
        elif tool_name == "edit_file":
            return SkillResult(output=self._edit_file(args))
        return SkillResult(output=f"Unknown tool: {tool_name}", success=False)

    def _write_diary(self, args: dict) -> str:
        entry = (args.get("entry") or "").strip()
        if not entry:
            return "Error: entry is required."
        return diary.append(entry, source="chat", section="今日日記")

    def _read_diary(self, args: dict) -> str:
        date_str = (args.get("date") or "").strip()
        if not date_str:
            content = diary.read_raw()
            return content if content else "Today's diary is empty."

        try:
            year_month = date_str[:7]
            archive_dir = diary.path.parent / "diary_archive"
            archive_path = archive_dir / f"{year_month}.md"
            if not archive_path.exists():
                return f"No diary archive found for {year_month}."

            raw = archive_path.read_text(encoding="utf-8")
            lines = raw.split("\n")
            collecting = False
            result: list[str] = []
            for line in lines:
                if line.startswith("# Diary ") and date_str in line:
                    collecting = True
                    result.append(line)
                elif collecting and line.startswith("# Diary "):
                    break
                elif collecting:
                    result.append(line)

            if not result:
                return f"No diary entry found for {date_str}."
            return "\n".join(result).strip()
        except Exception as e:
            return f"Error reading diary archive: {e}"

    def _edit_file(self, args: dict) -> str:
        action = (args.get("action") or "").lower()
        rel_path = (args.get("path") or "").strip()

        if not rel_path:
            return "Error: path is required."
        if not rel_path.endswith(".md"):
            return "Error: only .md files are supported."

        target = (_DATA_DIR / rel_path).resolve()
        if not str(target).startswith(str(_DATA_DIR.resolve())):
            return "Error: path must be within data/ directory."

        if action == "read":
            if not target.exists():
                return f"File not found: {rel_path}"
            return target.read_text(encoding="utf-8")

        elif action == "write":
            content = args.get("content")
            if content is None:
                return "Error: content is required for write."
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".md.tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(target)
            log.info("edit_file: wrote %s (%d chars)", rel_path, len(content))
            return f"OK: {rel_path} written ({len(content)} chars)."

        return f"Error: unknown action '{action}'. Use read or write."
