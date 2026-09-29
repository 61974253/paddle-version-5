from paddleocr import PaddleOCR

print("Starting PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

print("PaddleOCR initialized successfully!")
