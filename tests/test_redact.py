from screex.core import redact


def test_redacts_email():
    out, kinds = redact.redact_line("contact rushi@acme.io for help")
    assert "rushi@acme.io" not in out
    assert "email" in kinds
    assert "[REDACTED:email]" in out


def test_redacts_api_key_prefix():
    out, kinds = redact.redact_line("key: sk-live-9f2a7c4d8e1b2a3c")
    assert "sk-live-9f2a7c4d8e1b2a3c" not in out
    assert "api_key" in kinds


def test_redacts_aws_key():
    out, kinds = redact.redact_line("AKIAIOSFODNN7EXAMPLE in config")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "aws_key" in kinds


def test_redacts_high_entropy_token():
    out, kinds = redact.redact_line("token aZ9k2Lp7Qw3Xr8Tn5Vb1Yc4 done")
    assert "aZ9k2Lp7Qw3Xr8Tn5Vb1Yc4" not in out
    assert "secret" in kinds


def test_redacts_credit_card_luhn():
    out, kinds = redact.redact_line("card 4111 1111 1111 1111 ok")
    assert "credit_card" in kinds
    assert "4111 1111 1111 1111" not in out


def test_ignores_non_luhn_digit_run():
    # 16 digits that fail the Luhn check should not be flagged as a card
    out, kinds = redact.redact_line("order 1234 5678 9012 3456 placed")
    assert "credit_card" not in kinds


def test_clean_line_unchanged():
    out, kinds = redact.redact_line("Open Settings and click Save")
    assert out == "Open Settings and click Save"
    assert kinds == []


def test_redacts_multiple_distinct_secrets_in_order():
    # Two different secrets on one line: both masked, left-to-right, plain text preserved.
    out, kinds = redact.redact_line("email rushi@acme.io key sk-live-9f2a7c4d8e1b2a3c")
    assert out == "email [REDACTED:email] key [REDACTED:api_key]"
    assert kinds == ["email", "api_key"]


def test_overlapping_matches_collapse_to_one_span():
    # A token that matches BOTH the api_key pattern and the high-entropy heuristic must be
    # reported once — find_secrets keeps the earliest, longest span — not double-masked.
    text = "token ghp_aZ9k2Lp7Qw3Xr8Tn5Vb1Yc4 here"
    spans = redact.find_secrets(text)
    assert len(spans) == 1
    assert spans[0][2] == "api_key"
    out, kinds = redact.redact_line(text)
    assert kinds == ["api_key"]
    assert out.count("[REDACTED:") == 1


def test_has_secret():
    assert redact.has_secret("email me at a@b.com")
    assert not redact.has_secret("just a normal sentence")
