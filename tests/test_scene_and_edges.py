from physical_ai_lab.config import load_config
from physical_ai_lab.edges import detect_edges_and_lines
from physical_ai_lab.matching import match_orb
from physical_ai_lab.scene import crop_and_resize, make_scene


def test_scene_shape_and_crop():
    config = load_config()
    frame = make_scene(config.width, config.height)
    views = crop_and_resize(frame)
    assert frame.shape == (config.height, config.width, 3)
    assert frame.dtype.name == "uint8"
    assert views["crop"].shape[0] == 200
    assert views["half"].shape == (config.height // 2, config.width // 2, 3)


def test_edges_find_lines():
    config = load_config()
    frame = make_scene(config.width, config.height)
    result = detect_edges_and_lines(frame)
    assert result["edges"].shape == (config.height, config.width)
    assert result["line_count"] > 0


def test_shifted_frames_share_orb_matches():
    config = load_config()
    frame0 = make_scene(config.width, config.height)
    frame1 = make_scene(config.width, config.height, config.scene_shift_x, config.scene_shift_y)
    matched = match_orb(frame0, frame1, nfeatures=config.orb_features, ratio=config.match_ratio)
    assert len(matched["keypoints0"]) > 0
    assert len(matched["matches"]) > 0
