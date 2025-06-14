# client.py

import socket
from PIL import Image
import struct
import os 
import argparse
os.makedirs('img', exist_ok=True)

parser = argparse.ArgumentParser(description="Select image mode")
parser.add_argument('--mode',"-m", choices=['rgb', 'yuv'], default='rgb', help="Image mode (rgb or yuv)")
args = parser.parse_args()


def yuv422_to_rgb888(y, u, v):
    """Convert YUV422 values to RGB888 tuple."""
    r = y + 1.402 * (v - 128)
    g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = y + 1.772 * (u - 128)
    return (int(max(0, min(255, r))),
            int(max(0, min(255, g))),
            int(max(0, min(255, b))))

def create_image_from_yuv422(data, width, height):
    """Create a PIL Image from YUV422 binary data."""
    pixels = []
    for i in range(0, len(data), 4):
        y0, u, y1, v = struct.unpack_from('BBBB', data, i)
        pixels.append(yuv422_to_rgb888(y0, u, v))
        pixels.append(yuv422_to_rgb888(y1, u, v))

    # Create an image from RGB data
    img = Image.new('RGB', (width, height))
    img.putdata(pixels)
    return img

def create_gray_from_yuv422(data, width, height):
    """Create a PIL Image from YUV422 binary data."""
    pixels = []
    for i in range(0, len(data), 2):
        y, c = struct.unpack_from('BB', data, i)
        pixels.append(y)

    # Create an image from RGB data
    img = Image.new('L', (width, height))
    img.putdata(pixels)
    return img


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

icnt = 0

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

    if args.mode == 'yuv':
        create_image = create_image_from_yuv422
    else:
        create_image = create_image_from_rgb565

    imgrgb = create_image(result_bytes,176,144)
    imgrgb.save(f'img/img_{icnt:04d}.png')
    if args.mode == 'yuv':
        imggray = create_gray_from_yuv422(result_bytes,176,144)
        imggray.save(f'img/gray_{icnt:04d}.png')
        
    icnt += 1
    print(f"Image saved as img/img_{icnt:04d}.png")
    # imgrgb.show()

