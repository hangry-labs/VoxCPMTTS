from voxcpm.timestamps.base import TimestampItem
from voxcpm.timestamps.stable_ts import extract_timestamp_items, split_word_items_to_chars


def test_split_word_items_to_chars_evenly_distributes_word_duration():
    chars = split_word_items_to_chars(
        [TimestampItem(text="welcome", start=0.5, end=1.2, level="word")]
    )

    assert [item.text for item in chars] == list("welcome")
    assert chars[0].start == 0.5
    assert chars[-1].end == 1.2
    assert all(item.level == "char" for item in chars)


def test_extract_timestamp_items_supports_mapping_results():
    result = {
        "segments": [
            {
                "text": "Hello world",
                "start": 0.1,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.1, "end": 0.5},
                    {"word": "world", "start": 0.55, "end": 1.0},
                ],
            }
        ]
    }

    words = extract_timestamp_items(result, "word")

    assert [item.text for item in words] == ["Hello", "world"]
    assert words[1].end == 1.0
