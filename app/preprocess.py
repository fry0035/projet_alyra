import numpy as np
from PIL import Image
import io


def prepocess_image(image_bytes):
    # Convert the image to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Resize the image to 28x28 pixels
    image = image.resize((600, 600))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Reshape the array to match the input shape of the model (1, 28, 28, 1)
    image_final = np.reshape(image_array, (1, 600, 600, 3))

    return image_final
