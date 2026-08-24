from regnav.benchmark import _image_shape
from regnav.config import ModelConfig


def test_image_shape_includes_temporal_context():
    config = ModelConfig(image_size=(28, 42), context_frames=4)

    assert _image_shape(config) == (1, 3, 4, 42, 28)

