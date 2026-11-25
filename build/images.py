"""Writing image files: raw bytes, JPEG re-encodings and rendered PDF pages."""
import io
import os
import shutil


def suffix_of(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"BM"):
        return ".bmp"
    raise ValueError("unknown image format")


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def place_file(src, dst, link=False):
    """Copy a file of the release, or hard-link it when asked and possible."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)


def jpeg(data, quality):
    """Re-encode an image as RGB JPEG, as the evaluation copies of EMMA images were made."""
    from PIL import Image
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()


def render_page(document, page_number, dpi=72, quality=90):
    """Render one page (counted from 1) of an open PDF document as a JPEG."""
    from PIL import Image
    png = document[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
    out = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()
