# client.py

import socket
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


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
soc.connect(("192.168.4.1", 3333))

while True:
    result_bytes = soc.recv(4) # the number means how the response can be in bytes
    print("RAW Length:",result_bytes)
    flen = int.from_bytes(result_bytes, 'big')
    print("Length:",flen)

    result_bytes = bytearray()
    while len(result_bytes) < flen:
        chunk = soc.recv(flen - len(result_bytes))
        if not chunk:
            raise ConnectionError("Connection lost while receiving data")
        result_bytes.extend(chunk)

    imgrgb = create_image_from_rgb565(result_bytes,176,144)
    imgrgb.show()

imgrgb.save('output.png')

with open("img.raw","wb") as f:
    f.write(result_bytes)


# Example usage
#with open('image_rgb565.raw', 'rb') as f:
#    raw_data = f.read()

