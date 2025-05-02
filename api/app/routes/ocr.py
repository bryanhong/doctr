# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Response

from app.schemas import OCRBlock, OCRIn, OCRLine, OCROut, OCRPage, OCRWord
from app.utils import get_documents, resolve_geometry
from app.vision import init_predictor
import pickle
import torch

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

#    results = [
#        OCROut(
#            name=filenames[i],
#            orientation=page.orientation,
#            language=page.language,
#            dimensions=page.dimensions,
#            items=[
#                OCRPage(
#                    blocks=[
#                        OCRBlock(
#                            geometry=resolve_geometry(block.geometry),
#                            objectness_score=round(block.objectness_score, 2),
#                            lines=[
#                                OCRLine(
#                                    geometry=resolve_geometry(line.geometry),
#                                    objectness_score=round(line.objectness_score, 2),
#                                    words=[
#                                        OCRWord(
#                                            value=word.value,
#                                            geometry=resolve_geometry(word.geometry),
#                                            objectness_score=round(word.objectness_score, 2),
#                                            confidence=round(word.confidence, 2),
#                                            crop_orientation=word.crop_orientation,
#                                        )
#                                        for word in line.words
#                                    ],
#                                )
#                                for line in block.lines
#                            ],
#                        )
#                        for block in page.blocks
#                    ]
#                )
#            ],
#        )
#        for i, page in enumerate(out.pages)
#    ]
#
#    return results

    # This returns a list of tuples (filename, page) that OCRmyPDF can use to apply hOCR to a PDF
    # Each page contains a list of blocks, each block contains a list of lines
    # Each line contains a list of words
    # Each word contains a value, geometry, objectness_score, confidence, and crop_orientation
    # Below we're exporting the results as hOCR, serializing it into binary bytes that the caller can
    # consume to apply hOCR to a PDF
    hOCR_list = out.export_as_xml()
    hOCR_serial = pickle.dumps(hOCR_list)
    # Empty the CUDA cache to free up memory
    # This is important because the model is loaded in GPU memory and we need to free it up for other apps
    torch.cuda.empty_cache()
    return Response(content=hOCR_serial, media_type="application/octet-stream")