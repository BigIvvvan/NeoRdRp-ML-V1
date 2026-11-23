from neordrp_ml_v1.embedding import clean_sequence, plan_windows


def test_clean_sequence_and_windows():
    assert clean_sequence(" ac-d*ez ") == "ACDEX"
    assert plan_windows(100, "window") == [(0, 100)]
    windows = plan_windows(3000, "window", maximum=2048, stride=512)
    assert windows[0] == (0, 2048)
    assert windows[-1] == (952, 3000)
