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
        return "[Error] pytesseract is not installed."

    # Try to find Tesseract if not in PATH (common on Windows)
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        import os
        import platform
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.join(os.environ.get("USERPROFILE", ""), r"AppData\Local\Tesseract-OCR\tesseract.exe")
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        if "tesseract is not installed or it's not in your PATH" in str(e).lower():
            return "[Error] Tesseract-OCR is not installed on the system. Please install it from https://github.com/UB-Mannheim/tesseract/wiki"
        return f"[Error] OCR failed: {str(e)}"


def extract_text_from_pdf(uploaded_file) -> str:
    if fitz is None:
        return "[Error] PyMuPDF (fitz) is not installed. PDF processing is unavailable."
    try:
        uploaded_file.seek(0)
        pdf_stream = io.BytesIO(uploaded_file.read())
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        all_text = []
        is_scanned = True
        
        # First pass: try to extract text normally
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                all_text.append(page_text)
                # If we find a significant amount of text, it's likely not a scanned-only PDF
                if len(page_text) > 50:
                    is_scanned = False
        
        extracted_text = "\n\n".join(all_text).strip()
        
        # If we found enough text and it doesn't look like a scan, return it
        if extracted_text and not is_scanned:
            return extracted_text

        # Fallback: Perform OCR on all pages if it looks like a scanned PDF or no text was found
        ocr_text_parts = []
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            page_ocr = extract_text_from_image(img)
            if page_ocr:
                ocr_text_parts.append(page_ocr)
        
        final_text = "\n\n".join(ocr_text_parts).strip()
        
        if not final_text and not extracted_text:
            return "[Warning] No text could be extracted. The PDF might be empty or Tesseract-OCR might not be installed on the system."
            
        return final_text if len(final_text) > len(extracted_text) else extracted_text
        
    except Exception as e:
        return f"[Error] Failed to extract text from PDF: {str(e)}"
    finally:
        if 'doc' in locals():
            doc.close()


def extract_text_from_file(uploaded_file) -> str:
    if is_pdf_file(uploaded_file.name):
        return extract_text_from_pdf(uploaded_file)
    try:
        image = load_image_from_bytes(uploaded_file)
        return extract_text_from_image(image)
    except Exception:
        return ""
