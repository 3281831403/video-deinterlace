"""Red tests for the field-based deinterlacing kernels.

The tests are intentionally read-only: fix ``deint/fields.py`` instead of
editing this file.  Every test currently fails with ``NotImplementedError``
from one of the eight stubs.  Frames are ``list[list[float]]`` grayscale
images and numeric comparisons use an absolute tolerance of ``1e-6``.
"""

import pytest

from deint.fields import (
    DeintError,
    adaptive_deint,
    blend_fields,
    bob,
    deint_batch,
    motion_detect,
    split_fields,
    validate_frame,
    weave,
)


def _constant(h, w, value):
    return [[float(value)] * w for _ in range(h)]


def _assert_frame_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for row, expected_row in zip(actual, expected):
        assert len(row) == len(expected_row)
        assert row == pytest.approx(expected_row, abs=tol)


def test_split_fields_starts_at_row_zero():
    source = [[0.0], [0.1], [0.2], [0.3]]

    top, bottom = split_fields(source, "top")

    _assert_frame_close(top, [[0.0], [0.2]])
    _assert_frame_close(bottom, [[0.1], [0.3]])


def test_split_fields_honours_field_order():
    source = [[0.1], [0.2], [0.3], [0.4]]

    top_first = split_fields(source, "top")
    bottom_first = split_fields(source, "bottom")

    _assert_frame_close(top_first[0], [[0.1], [0.3]])
    _assert_frame_close(top_first[1], [[0.2], [0.4]])
    _assert_frame_close(bottom_first[0], [[0.2], [0.4]])
    _assert_frame_close(bottom_first[1], [[0.1], [0.3]])


def test_weave_restores_row_order():
    top = [[0.1], [0.3], [0.5]]
    bottom = [[0.2], [0.4], [0.6]]

    result = weave(top, bottom, "top")

    _assert_frame_close(
        result,
        [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
    )


def test_bob_interpolates_missing_lines():
    field = [[0.0], [0.5], [1.0]]

    result = bob(field, "top")

    assert len(result) == 6
    _assert_frame_close(
        result,
        [[0.0], [0.25], [0.5], [0.75], [1.0], [1.0]],
    )


def test_motion_detect_uses_threshold_in_dark_areas():
    top = [[0.01, 0.02, 0.03]]
    bottom = [[0.005, 0.055, 0.06]]

    mask = motion_detect(top, bottom, 0.01)

    _assert_frame_close(mask, [[0.0, 1.0, 1.0]])


def test_adaptive_deint_bob_interpolates_motion_lines():
    frame = [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.8, 0.2],
    ]

    result = adaptive_deint(frame, 0.1)

    _assert_frame_close(
        result,
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.2, 0.8],
        ],
    )


def test_odd_height_frame_raises():
    frame = [[0.1], [0.2], [0.3]]

    issues = validate_frame(frame)
    assert any("odd" in issue.lower() for issue in issues)
    with pytest.raises(DeintError):
        split_fields(frame, "top")


def test_nan_pixel_is_rejected():
    bad = [[0.1, 0.2], [0.3, float("nan")]]

    issues = validate_frame(bad)
    assert any("finite" in issue.lower() for issue in issues)
    with pytest.raises(DeintError):
        split_fields(bad, "top")
    with pytest.raises(DeintError):
        adaptive_deint(bad, 0.1)
    with pytest.raises(DeintError):
        deint_batch([bad], {"threshold": 0.1})


def test_batch_keeps_frame_buffers_independent_and_rolls_back():
    frame_a = _constant(4, 2, 0.1)
    frame_b = _constant(4, 2, 0.9)

    result = deint_batch([frame_a, frame_b], {"threshold": 0.1})
    out_a, out_b = result["frames"]

    assert result["count"] == 2
    _assert_frame_close(out_a, frame_a)
    _assert_frame_close(out_b, frame_b)

    out_a[0][0] = 0.0
    assert out_b[0][0] == pytest.approx(0.9, abs=1e-6)
    assert frame_a[0][0] == pytest.approx(0.1, abs=1e-6)

    bad = [[0.5, 0.5], [0.2, 0.2], [0.5, 0.5]]
    with pytest.raises(DeintError):
        deint_batch([frame_a, bad], {"threshold": 0.1})
    assert frame_a[0][0] == pytest.approx(0.1, abs=1e-6)


def test_threshold_out_of_range_raises():
    with pytest.raises(DeintError):
        motion_detect([[0.0]], [[1.0]], -0.01)
    with pytest.raises(DeintError):
        motion_detect([[0.0]], [[1.0]], 1.01)
    with pytest.raises(DeintError):
        motion_detect([[0.0]], [[1.0]], float("nan"))
    with pytest.raises(DeintError):
        adaptive_deint([[0.0], [1.0]], 2.0)
    with pytest.raises(DeintError):
        deint_batch([], {"threshold": -1.0})


def test_static_adaptive_matches_weave():
    frame = [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.3, 0.4],
        [0.3, 0.4],
    ]

    top, bottom = split_fields(frame, "top")
    expected = weave(top, bottom, "top")
    actual = adaptive_deint(frame, 0.01)

    _assert_frame_close(actual, expected)


def test_blend_preserves_original_row_count():
    frame = [
        [0.0, 0.2],
        [0.4, 0.6],
        [0.8, 1.0],
        [0.2, 0.4],
    ]

    top, bottom = split_fields(frame, "top")
    blended = blend_fields(top, bottom)

    assert len(blended) == len(frame)
    assert all(len(row) == len(frame[0]) for row in blended)
    _assert_frame_close(
        blended,
        [
            [0.2, 0.4],
            [0.2, 0.4],
            [0.5, 0.7],
            [0.5, 0.7],
        ],
    )


def test_single_row_frame_raises_deinterror():
    with pytest.raises(DeintError):
        split_fields([[0.25, 0.75]], "top")


def test_empty_frame_returns_empty():
    assert split_fields([], "top") == ([], [])
    assert weave([], [], "top") == []
    assert bob([], "top") == []
    assert blend_fields([], []) == []
    assert motion_detect([], [], 0.1) == []
    assert adaptive_deint([], 0.1) == []
    assert validate_frame([]) == ["frame is empty"]

    result = deint_batch([], {"threshold": 0.1})

    assert result == {"frames": [], "count": 0}


def test_batch_success_preserves_input_order():
    values = [0.05, 0.15, 0.25, 0.35, 0.45]
    frames = [_constant(4, 2, value) for value in values]

    result = deint_batch(frames, {"threshold": 0.1})
    processed = result["frames"]

    assert result["count"] == 5
    assert [frame[0][0] for frame in processed] == pytest.approx(
        values, abs=1e-6
    )
    assert all(len(frame) == 4 and len(frame[0]) == 2 for frame in processed)
