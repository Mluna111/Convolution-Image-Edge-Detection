from PIL import Image, ImageOps
import numpy as np


def convolve(image, kernel):
    """
    Convolves the image with the given kernel.
    """
    # Pad the image to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant')
    output_images = Image.fromarray(padded_image)
    output_images.save("padded.png")

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Create an output image
    output_image = np.zeros_like(image)

    # Convolve the image with the kernel
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                output_image[y, x, c] = np.sum(padded_image[y:y + 3, x:x + 3, c] * kernel)

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image


def detect_edges(image, kernel):
    """
    Detects edges in the image using the given kernel.
    """
    # Convert the image to a NumPy array
    image_array = np.array(image)
    print(f"Image array shape: {image_array.shape}")

    # Convolve the image with the kernel
    edge_image_array = convolve(image_array, kernel)
    Image.fromarray(edge_image_array).save("after_convolution.png")

    # Normalize the output image
    min_val = edge_image_array.min()
    max_val = edge_image_array.max()
    print(f"Min value after convolution: {min_val}, Max value after convolution: {max_val}")

    if max_val - min_val != 0:
        edge_image_array = (edge_image_array - min_val) / (max_val - min_val) * 255
    else:
        edge_image_array = np.zeros_like(edge_image_array)

    edge_image_array = edge_image_array.astype(np.uint8)

    # Convert the NumPy array back to a PIL Image
    edge_image = Image.fromarray(edge_image_array)

    return edge_image


# Define the kernel (you can modify the values)

# Edge Detection
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Sharpen Image
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])

# Load the image
image = Image.open('super.png')
image = image.convert('RGB')  # Ensure the image is in RGB mode
image.save("original.png")

# Detect edges using the kernel
edges = detect_edges(image, kernel)
edges.save('reconstructed_super.png')

# Display the edge image
edges.show()


