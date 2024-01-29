import tensorflow as tf
import numpy as np
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray

mnist_runner = bentoml.tensorflow.get("tensorflow_mnist:latest").to_runner()

svc = bentoml.Service(
    name="tensorflow_mnist_hypertuned",
    runners=[mnist_runner],
)

@svc.api(input=Image(), output=NumpyNdarray(dtype="float32"))
async def predict_image(f: PILImage) -> "np.ndarray":
    assert isinstance(f, PILImage)
    arr = np.array(f) / 255.0
    print(f'arr shape: {arr.shape}')
    gray_arr = tf.image.rgb_to_grayscale(arr)
    print(f'arr shape: {gray_arr.shape}')
    final_arr = np.squeeze(gray_arr)
    print(f"final array: {final_arr.shape}")
    assert final_arr.shape == (28, 28)

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension
    arr = np.expand_dims(final_arr, (0,3)).astype("float32")  # reshape to [1, 28, 28, 1]
    print(f'after expanding dimensions: {arr.shape}')
    return await mnist_runner.async_run(arr)
