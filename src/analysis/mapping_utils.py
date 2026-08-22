import base64
import folium
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def process_logo_to_clean_favicon(logo_path: Path, size: tuple = (64, 64)) -> str:
    """
    Downsamples the INSPYRE logo to standard favicon size, trims white 
    background and bottom text, and returns a compact Base64 PNG string.
    """
    img = Image.open(logo_path).convert("RGBA")

    # 1. Resize to standard favicon dimensions to keep Base64 string tiny
    img = img.resize(size, Image.Resampling.LANCZOS)
    width, height = img.size

    # 2. Create circular mask to slice off the bottom arc text
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    center_x, center_y = width / 2, height / 2 - 2
    radius = min(width, height) / 2 - 2
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)

    # 3. Apply numpy vectorized transparency mask
    data = np.array(img)

    cleaned_img = Image.fromarray(data, mode="RGBA")

    # 4. Save to buffer and encode
    buffered = io.BytesIO()
    cleaned_img.save(buffered, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64_str}"


def add_local_png_favicon(map_obj: folium.Map, logo_path: str):
    """Injects the cleaned local PNG logo as the browser favicon in Folium."""
    logo_file = Path(logo_path)
    if not logo_file.exists():
        print(f"[-] Logo file not found at {logo_path}, skipping favicon injection.")
        return

    try:
        clean_b64_icon = process_logo_to_clean_favicon(logo_file, size=(64, 64))
        
        # Standard favicon and apple-touch-icon links for Chrome/Safari compatibility
        favicon_html = f'''
        <link rel="icon" type="image/png" sizes="64x64" href="{clean_b64_icon}">
        <link rel="shortcut icon" type="image/png" href="{clean_b64_icon}">
        '''
        map_obj.get_root().header.add_child(folium.Element(favicon_html))
        print("[+] Successfully injected INSPYRE favicon into map header.")
        return map_obj
    except Exception as e:
        print(f"[-] Failed to process favicon: {e}")
        return map_obj


def encode_image_to_base64(img_path: Path) -> str:
    """Encodes a JPEG plot image to a base64 string for embedding into HTML popups."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")