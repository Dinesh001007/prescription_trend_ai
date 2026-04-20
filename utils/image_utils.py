import io

try:
    from PIL import Image
except ImportError:
    Image = None


try:
    import fitz
except ImportError:
    fitz = None


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))


def is_pdf_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def load_image_from_bytes(uploaded_file) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow is required to process image uploads. Install it via pip install Pillow.")
    uploaded_file.seek(0)
    return Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")


def extract_text_from_image(image) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""

    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""


def extract_text_from_pdf(uploaded_file) -> str:
    if fitz is None:
        return ""
    try:
        uploaded_file.seek(0)
        pdf_stream = io.BytesIO(uploaded_file.read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text_lines = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_lines.append(page_text)
        extracted_text = "\n".join(text_lines).strip()
        if extracted_text:
            return extracted_text

        # Fall back to OCR on the first page image if the PDF contains a scan
        if doc.page_count > 0:
            page = doc[0]
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            return extract_text_from_image(image)
    except Exception:
        return ""
    return ""


def extract_text_from_file(uploaded_file) -> str:
    if is_pdf_file(uploaded_file.name):
        return extract_text_from_pdf(uploaded_file)
    try:
        image = load_image_from_bytes(uploaded_file)
        return extract_text_from_image(image)
    except Exception:
        return ""
