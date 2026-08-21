from PIL import Image
import torch

from regnav.inference import preprocess_image


def test_preprocess_image_uses_model_size_and_imagenet_normalization():
    image = Image.new("RGB", (8, 6), (0, 0, 0))

    tensor = preprocess_image(image, (32, 18))

    assert tensor.shape == (3, 18, 32)
    torch.testing.assert_close(
        tensor[:, 0, 0],
        torch.tensor([-2.1179, -2.0357, -1.8044]),
        rtol=1e-4,
        atol=1e-4,
    )
