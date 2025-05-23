# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Response

from app.schemas import OCRBlock, OCRIn, OCRLine, OCROut, OCRPage, OCRWord
from app.utils import get_documents, resolve_geometry
from app.vision import init_predictor
import pickle
import torch

from doctr.io import DocumentFile
from tempfile import TemporaryDirectory
from ocrmypdf.hocrtransform import HocrTransform
from PIL import Image
import os
from PyPDF2 import PdfMerger, PdfReader, PdfWriter

router = APIRouter()

@router.post("/", response_model=list[OCROut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(request: OCRIn = Depends(), files: list[UploadFile] = [File(...)]):
    """Runs docTR OCR model to analyze the input image"""
    try:
        # generator object to list
        content, filenames = await get_documents(files)
        predictor = init_predictor(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    out = predictor(content)

    xml_outputs = out.export_as_xml()
    # Empty the CUDA cache to free up memory
    # This is important because the model is loaded in GPU memory and we need to free it up for other apps
    torch.cuda.empty_cache()
    await files[0].seek(0)
    file_bytes = await files[0].read()
    # Scanned images come in at 300 DPI, ensure we tell pypdfium2 (Documentfile uses it) to render at that DPI
    dpi = 300
    scale = dpi / 72.0
    docs = DocumentFile.from_pdf(file_bytes, scale=scale)
    pdf_paths = []

    with TemporaryDirectory() as tmpdir: # delete=False can be set to keep the files
        for i, (xml, img) in enumerate(zip(xml_outputs, docs)):
            img_path = os.path.join(tmpdir, f"{i}.jpg")
            xml_path = os.path.join(tmpdir, f"{i}.xml")
            pdf_path = os.path.join(tmpdir, f"{i}.pdf")

            Image.fromarray(img).save(img_path, format="JPEG", quality=90)

            with open(xml_path, "w") as f:
                f.write(xml[0].decode())

            hocr = HocrTransform(hocr_filename=xml_path, dpi=300)
            hocr.to_pdf(out_filename=pdf_path, image_filename=img_path)

            pdf_paths.append(pdf_path)

        # Merge all PDFs into one
        merged_pdf_path = os.path.join(tmpdir, "merged.pdf")
        merger = PdfMerger()
        for pdf in sorted(pdf_paths):  # Ensure consistent page order
            merger.append(pdf)
        merger.write(merged_pdf_path)
        merger.close()

        reader = PdfReader(merged_pdf_path)
        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        metdata = reader.metadata or {}
        metdata.update(
            {
                "/Producer": "docTR",
                "/Creator": "docTR",
            }
        )
        writer.add_metadata(metdata)
        with open(merged_pdf_path, "wb") as f:
            writer.write(f)

        # Read and return the merged PDF
        with open(merged_pdf_path, "rb") as f:
            merged_bytes = f.read()

        return Response(content=merged_bytes, media_type="application/pdf")

# pip install git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git
# apt-get install tesseract-ocr ghostscript
# ocrmypdf --force-ocr --pdf-renderer=sandwich --language=eng merged.pdf final.pdf
