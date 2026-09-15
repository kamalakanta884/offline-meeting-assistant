from src.redaction.redact import redact_text


def test_redacts_id_number():
    text = "Aadhaar 1234 5678 9012 was shared by mistake"
    result = redact_text(text)
    assert "1234 5678 9012" not in result
    assert "REDACTED" in result


def test_leaves_clean_text_untouched():
    text = "Meeting notes with no sensitive info here."
    assert redact_text(text) == text


def test_no_corruption_on_overlapping_matches():
    text = "My account number is 123456789012 and phone is 9876543210"
    result = redact_text(text)
    # A broken interleaved replace would produce mangled/duplicated markers
    # like "[REDACTED:phone_number]count_number]" instead of clean ones.
    assert "123456789012" not in result
    assert "9876543210" not in result
    assert result.count("[REDACTED:") == 2
