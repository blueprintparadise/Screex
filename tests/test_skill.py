from screex import skill


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
