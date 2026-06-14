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
