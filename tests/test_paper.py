from thermalview import paper


def test_fill_reports_missing_and_never_invents():
    text, missing = paper.fill("MAE {{a}} and {{ b }}; {{c}}", {"a": "1.23", "b": 4})
    assert text == "MAE 1.23 and 4; [[missing: c]]" and missing == ["c"]


def test_direction_words_follow_the_interval():
    assert paper._sig(0.1, 0.3) == "better than"
    assert paper._sig(-0.3, -0.1) == "worse than"
    assert paper._sig(-0.1, 0.2) == "not distinguishable from"
    assert paper._sig(float("nan"), float("nan")) == "not distinguishable from"


def test_citations_numbered_by_first_appearance():
    from thermalview.paper import cite
    refs = {"a": {"text": "A.", "verified": True}, "b": {"text": "B.", "verified": False}}
    text, issues = cite("x [@b] y [@a; @b] z\n{{REFERENCES}}", refs)
    assert text.startswith("x [1] y [1, 2] z")
    assert "[1] B." in text and "[2] A." in text
    assert any("b" in i for i in issues)
    text, issues = cite("[@nope]", refs)
    assert "unknown ref" in text and "nope" in issues
