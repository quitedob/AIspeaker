"""Load and manage skill definitions from the skills/ directory."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def load_skills() -> dict[str, dict]:
    """Load all skill JSON files from skills/ directory.
    Returns dict mapping skill_id -> skill data.
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skills = {}
    for f in sorted(SKILLS_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            skill_id = f.stem
            data["id"] = skill_id
            skills[skill_id] = data
            logger.info(f"Loaded skill: {skill_id} - {data.get('name', '?')}")
        except Exception as e:
            logger.warning(f"Failed to load skill {f}: {e}")
    logger.info(f"Total skills loaded: {len(skills)}")
    return skills


def get_skill_prompt(skills: dict[str, dict], skill_id: str, language: str = "zh") -> str:
    """Get the system prompt for a specific skill and language."""
    if not skill_id or skill_id not in skills:
        return ""
    skill = skills[skill_id]
    prompts = skill.get("prompts", {})
    return prompts.get(language, prompts.get("zh", ""))


def get_skills_list(skills: dict[str, dict]) -> list[dict]:
    """Return a simplified list for the frontend dropdown."""
    result = []
    for sid, data in skills.items():
        result.append({
            "id": sid,
            "name": data.get("name", sid),
            "icon": data.get("icon", "🎭"),
            "description": data.get("description", ""),
        })
    return result
