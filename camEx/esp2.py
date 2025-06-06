# client.py

from PIL import Image
import struct


def rgb565_to_rgb888(value):
    """Convert a single 16-bit RGB565 value to 24-bit RGB888 tuple."""
    r = (value >> 11) & 0x1F
    g = (value >> 5) & 0x3F
    b = value & 0x1F

    # Scale values up to 8-bit range
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)

    return (r, g, b)

def create_image_from_rgb565(data, width, height):
    """Create a PIL Image from RGB565 binary data."""
    pixels = []
    for i in range(0, len(data), 2):
        # Read two bytes as one 16-bit RGB565 value
        pixel_value = struct.unpack_from('>H', data, i)[0]
        pixels.append(rgb565_to_rgb888(pixel_value))

    # Create an image from RGB data
    img = Image.new('RGB', (width, height))
    img.putdata(pixels)
    return img




# Example usage
with open('img.raw', 'rb') as f:
    raw_data = f.read()

imgrgb = create_image_from_rgb565(raw_data,176,144)
imgrgb.show()
