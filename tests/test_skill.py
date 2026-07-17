"""Tests for the cross-agent Agent Skill package."""

import os
import pytest


SKILL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills", "opennomark")
SKILL_PATH = os.path.join(SKILL_DIR, "SKILL.md")


class TestSkill:
    """Validate skill file structure and content."""

    def test_skill_file_exists(self):
        assert os.path.exists(SKILL_PATH), f"Skill file not found at {SKILL_PATH}"

    def test_has_frontmatter(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        assert content.startswith("---"), "Skill must start with YAML frontmatter"
        parts = content.split("---", 2)
        assert len(parts) >= 3, "Skill must have opening and closing --- for frontmatter"

    def test_frontmatter_has_name(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        frontmatter = content.split("---")[1]
        assert "name:" in frontmatter, "Frontmatter must contain 'name'"

    def test_frontmatter_has_description(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        frontmatter = content.split("---")[1]
        assert "description:" in frontmatter, "Frontmatter must contain 'description'"

    def test_description_has_trigger_keywords(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        frontmatter = content.split("---")[1].lower()
        # Should contain trigger keywords
        assert "watermark" in frontmatter or "水印" in frontmatter, \
            "Description should contain watermark-related trigger keywords"

    def test_body_has_usage_examples(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        body = content.split("---", 2)[2]
        assert "```" in body, "Skill body should contain code examples"
        assert "opennomark" in body, "Skill body should reference the CLI tool"

    def test_body_not_too_long(self):
        with open(SKILL_PATH) as f:
            lines = f.readlines()
        assert len(lines) < 500, f"Skill should be under 500 lines, got {len(lines)}"

    def test_skill_name_matches(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        frontmatter = content.split("---")[1]
        assert "opennomark" in frontmatter, "Skill name should be 'opennomark'"

    def test_uses_portable_cli_commands(self):
        with open(SKILL_PATH) as f:
            content = f.read()
        assert "uvx --from git+https://github.com/NanmiCoder/OpenNoMark.git" in content
        assert "--json" in content
        assert "/Users/" not in content

    def test_openai_interface_metadata_exists(self):
        metadata_path = os.path.join(SKILL_DIR, "agents", "openai.yaml")
        assert os.path.exists(metadata_path)
