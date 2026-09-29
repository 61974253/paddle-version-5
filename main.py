import os
import sys
import shutil
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_HTML_PATH = os.path.join(
    BASE_DIR,
    "index.html"
)


# ============================================================
# WINDOWS PADDLE DLL CONFIGURATION
# ============================================================

if sys.platform == "win32":

    paddle_dll_path = os.path.join(
        os.path.dirname(sys.executable),
        "Lib",
        "site-packages",
        "paddle",
        "libs"
    )

    if os.path.exists(paddle_dll_path):
        os.add_dll_directory(paddle_dll_path)


# ============================================================
# PADDLEOCR
# ============================================================

from paddleocr import PaddleOCR


print()
print("=" * 60)
print("Starting PaddleOCR PP-OCRv5...")
print("=" * 60)


ocr = PaddleOCR(
    lang="en",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)


print("PaddleOCR PP-OCRv5 initialized successfully!")
print("=" * 60)
print()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PP-OCRv5 OCR API",
    version="1.0.0",
    description="PaddleOCR PP-OCRv5 OCR Server"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def read_index():

    if not os.path.exists(INDEX_HTML_PATH):

        return {
            "success": False,
            "error": "index.html not found",
            "path": INDEX_HTML_PATH
        }

    return FileResponse(INDEX_HTML_PATH)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():

    return {
        "success": True,
        "status": "running",
        "engine": "PP-OCRv5",
        "paddleocr": "3.7.0"
    }


# ============================================================
# OCR ENDPOINT
# ============================================================

@app.post("/ocr")
async def process_ocr(
    file: UploadFile = File(...)
):

    safe_filename = os.path.basename(
        file.filename
    )

    temp_path = os.path.join(
        BASE_DIR,
        f"temp_{safe_filename}"
    )

    # Server-side processing timer
    server_start_time = time.perf_counter()

    try:

        # ====================================================
        # SAVE UPLOADED FILE
        # ====================================================

        with open(
            temp_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        print()
        print("=" * 60)
        print("OCR REQUEST RECEIVED")
        print("=" * 60)
        print(
            f"File: {safe_filename}"
        )


        # ====================================================
        # RUN PADDLEOCR
        # ====================================================

        results = ocr.predict(
            temp_path
        )


        extracted_data = []
        all_confidences = []


        # ====================================================
        # PROCESS RESULTS
        # ====================================================

        for res in results:

            data = res.json

            if callable(data):
                data = data()


            result_data = data.get(
                "res",
                {}
            )


            texts = result_data.get(
                "rec_texts",
                []
            )


            scores = result_data.get(
                "rec_scores",
                []
            )


            print(
                f"Detected text blocks: "
                f"{len(texts)}"
            )


            # =================================================
            # TEXT + CONFIDENCE
            # =================================================

            for i, text in enumerate(texts):

                if text is None:
                    continue


                text = str(
                    text
                ).strip()


                if not text:
                    continue


                score = 0.0


                if i < len(scores):

                    try:

                        score = float(
                            scores[i]
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        score = 0.0


                confidence = score * 100


                extracted_data.append({

                    "text": text,

                    "confidence":
                        round(
                            confidence,
                            2
                        )
                })


                all_confidences.append(
                    confidence
                )


        # ====================================================
        # OVERALL ACCURACY
        # ====================================================

        if all_confidences:

            overall_accuracy = (

                sum(
                    all_confidences
                )
                /
                len(
                    all_confidences
                )
            )

        else:

            overall_accuracy = 0.0


        # ====================================================
        # SERVER PROCESSING TIME
        # ====================================================

        server_elapsed_time = (
            time.perf_counter()
            -
            server_start_time
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success": True,

            "engine":
                "PP-OCRv5",

            "overall_accuracy":
                round(
                    overall_accuracy,
                    2
                ),

            "overall_accuracy_text":
                f"{overall_accuracy:.2f}%",

            "total_blocks":
                len(
                    extracted_data
                ),

            "processing_time_seconds":
                round(
                    server_elapsed_time,
                    2
                ),

            "processing_time_text":
                format_time(
                    server_elapsed_time
                ),

            "results":
                extracted_data
        }


        print(
            f"Overall accuracy: "
            f"{overall_accuracy:.2f}%"
        )

        print(
            f"Total blocks: "
            f"{len(extracted_data)}"
        )

        print(
            f"Processing time: "
            f"{format_time(server_elapsed_time)}"
        )

        print("=" * 60)
        print("OCR COMPLETED")
        print("=" * 60)
        print()


        return response


    except Exception as e:

        import traceback

        traceback.print_exc()


        server_elapsed_time = (
            time.perf_counter()
            -
            server_start_time
        )


        return {

            "success": False,

            "error":
                str(e),

            "overall_accuracy":
                0,

            "overall_accuracy_text":
                "0.00%",

            "total_blocks":
                0,

            "processing_time_seconds":
                round(
                    server_elapsed_time,
                    2
                ),

            "processing_time_text":
                format_time(
                    server_elapsed_time
                ),

            "results":
                []
        }


    finally:

        if os.path.exists(
            temp_path
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# TIME FORMAT FUNCTION
# ============================================================

def format_time(seconds):

    seconds = float(seconds)


    if seconds < 60:

        return (
            f"{seconds:.2f} seconds"
        )


    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds
        -
        (minutes * 60)
    )


    return (
        f"{minutes} min "
        f"{remaining_seconds:.2f} sec"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    print()
    print("=" * 60)
    print("PP-OCRv5 SERVER")
    print("=" * 60)

    print(
        "URL: http://127.0.0.1:8000"
    )

    print(
        "Health: http://127.0.0.1:8000/health"
    )

    print("=" * 60)
    print()


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
