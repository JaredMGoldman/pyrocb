import base64
import folium
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

def process_logo_to_clean_favicon(logo_path: Path, output_size: tuple = (128, 128)) -> str:
    """
    Tightly crops the INSPYRE logo, strips background padding, 
    and enhances contrast for high visibility in browser tabs.
    """
    img = Image.open(logo_path).convert("RGBA")
    
    # 1. Convert white/near-white outer background to transparent
    data = np.array(img)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    white_bg = (r > 230) & (g > 230) & (b > 230)
    data[..., 3] = np.where(white_bg, 0, a)
    
    clean_img = Image.fromarray(data, mode="RGBA")

    # 2. Tight bounding box crop to eliminate outer empty space
    bbox = clean_img.getbbox()
    if bbox:
        # Crop to contents
        clean_img = clean_img.crop(bbox)
        
        # Trim the bottom ~10% to eliminate the text arc
        w, h = clean_img.size
        clean_img = clean_img.crop((0, 0, w, int(h * 0.88)))

    # 3. Enhance contrast & sharpness for tiny icon legibility
    enhancer = ImageEnhance.Contrast(clean_img)
    clean_img = enhancer.enhance(1.2)
    sharpener = ImageEnhance.Sharpness(clean_img)
    clean_img = sharpener.enhance(1.4)

    # 4. Resize tightly into square dimensions
    clean_img = clean_img.resize(output_size, Image.Resampling.LANCZOS)

    # 5. Save optimized PNG buffer
    buffered = io.BytesIO()
    clean_img.save(buffered, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{b64_str}"


def add_local_png_favicon(map_obj: folium.Map, logo_path: str):
    """Injects the tight-cropped INSPYRE logo directly into the HTML <head>."""
    logo_file = Path(logo_path)
    if not logo_file.exists():
        print(f"[-] Logo file not found at {logo_path}, skipping favicon injection.")
        return

    try:
        clean_b64_icon = process_logo_to_clean_favicon(logo_file, output_size=(128, 128))
        
        favicon_html = f'''
        <link rel="icon" type="image/png" sizes="128x128" href="{clean_b64_icon}">
        <link rel="shortcut icon" type="image/png" href="{clean_b64_icon}">
        <link rel="apple-touch-icon" href="{clean_b64_icon}">
        '''
        map_obj.get_root().header.add_child(folium.Element(favicon_html))
        print("[+] Successfully injected enhanced INSPYRE favicon into <head>.")
        return map_obj
    except Exception as e:
        print(f"[-] Failed to process favicon: {e}")


def encode_image_to_base64(img_path: Path) -> str:
    """Encodes a JPEG plot image to a base64 string for embedding into HTML popups."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")