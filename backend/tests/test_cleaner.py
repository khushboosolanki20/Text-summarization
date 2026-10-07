from app.preprocessing.cleaner import clean_text, count_words, normalize_unicode


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text("   \n\n\t ") == ""


def test_unicode_ligatures_and_nbsp_are_normalised():
    assert normalize_unicode("efﬁcient model") == "efficient model"


def test_control_characters_are_removed():
    assert clean_text("Hello\x00 world\x07.") == "Hello world."


def test_hyphenated_line_break_is_joined():
    assert clean_text("text summari-\nzation works") == "text summarization works"


def test_hyphenated_proper_term_is_kept():
    # Next line starts with a capital/digit: a real hyphenated term, not a split word.
    assert clean_text("the COVID-\n19 pandemic") == "the COVID- 19 pandemic"


def test_hard_wrapped_lines_are_unwrapped_but_paragraphs_kept():
    raw = "First line of a\nwrapped paragraph.\n\nSecond paragraph\nhere."
    assert clean_text(raw) == "First line of a wrapped paragraph.\n\nSecond paragraph here."


def test_page_number_lines_are_removed():
    raw = "Some content here.\n12\nPage 3 of 10\n- 4 -\nMore content."
    assert clean_text(raw) == "Some content here. More content."


def test_bullets_become_separate_paragraphs():
    raw = "Key points:\n• First point is here\n• Second point is here"
    assert clean_text(raw) == "Key points:\n\nFirst point is here\n\nSecond point is here"


def test_whitespace_and_space_before_punctuation():
    assert clean_text("Too    many   spaces , here .") == "Too many spaces, here."


def test_windows_line_endings():
    assert clean_text("a line\r\nanother\r\n\r\nnew para") == "a line another\n\nnew para"


def test_count_words():
    assert count_words("one two  three\nfour") == 4
    assert count_words("") == 0
