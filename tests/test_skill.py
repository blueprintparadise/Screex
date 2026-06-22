from pathlib import Path

from screex import skill

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_and_distributed_skill_are_identical():
    """The pip-packaged skill (screex/SKILL.md) and the directory-distribution
    copy (skills/screex/SKILL.md, used by the Claude Skills Hub / plugin layout)
    must stay byte-identical so the two never drift."""
    packaged = _REPO_ROOT / "screex" / "SKILL.md"
    distributed = _REPO_ROOT / "skills" / "screex" / "SKILL.md"
    assert packaged.exists(), f"missing {packaged}"
    assert distributed.exists(), f"missing {distributed}"
    assert distributed.read_bytes() == packaged.read_bytes(), (
        "skills/screex/SKILL.md has drifted from screex/SKILL.md — "
        "copy the packaged file over it: cp screex/SKILL.md skills/screex/SKILL.md"
    )


def test_skill_text_is_bundled():
    text = skill.skill_text()
    assert text.startswith("---")
    assert "screex" in text


def test_install_skill_writes_file(tmp_path):
    target = skill.install_skill(tmp_path / "skills" / "screex")
    assert target.exists()
    assert target.name == "SKILL.md"
    assert target.read_text(encoding="utf-8") == skill.skill_text()


def test_skill_status_missing(tmp_path):
    assert skill.skill_status(tmp_path / "nope") == "missing"


def test_skill_status_current_after_install(tmp_path):
    dest = tmp_path / "s"
    skill.install_skill(dest)
    assert skill.skill_status(dest) == "current"


def test_skill_status_stale_when_content_differs(tmp_path):
    dest = tmp_path / "s"
    dest.mkdir()
    (dest / "SKILL.md").write_text("old content", encoding="utf-8")
    assert skill.skill_status(dest) == "stale"
