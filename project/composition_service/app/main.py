from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import numpy as np
import cv2
from app.service import CompositionService


app = FastAPI()
comp_service = CompositionService()


@app.post("/compose")
async def compose(foreground: UploadFile = File(...), mask: UploadFile = File(...), model_type: str = 'cnn', background_id: int = 1):
    """
    Composes a foreground image onto a selected background using a binary mask and harmonizes the result.
    
    Args:
        foreground (UploadFile): The uploaded foreground image.
        mask (UploadFile): The binary mask indicating foreground regions.
        model_type (str): The model type to use for harmonization ('none', 'cnn' or 'vit').
        background_id (int): The ID of the background image to use. Starts with 1

    Returns:
        StreamingResponse: A PNG image of the composed and harmonized result.
    
    Raises:
        HTTPException: If file types are invalid or image processing fails.
    """
    if not foreground.content_type.startswith('image/') or not mask.content_type.startswith('image/'):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload image files.")

    # read raw data
    foreground_data = await foreground.read()
    mask_data = await mask.read()

    try:
        # Decode images to OpenCV format
        foreground_np = cv2.imdecode(np.frombuffer(foreground_data, np.uint8), cv2.IMREAD_COLOR)
        mask_np = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask_np = (mask_np != 0).astype(np.uint8) * 255  # ensures mask is 0 or 255
        if foreground_np is None or mask_np is None:
            raise ValueError("Could not decode one or both images")

        res = comp_service.compose(foreground_np, mask_np, background_id, model_type)

        # Encode result as PNG
        success, encoded_img = cv2.imencode('.png', res)
        if not success:
            raise ValueError("Failed to encode result image")
        buf = BytesIO(encoded_img.tobytes())

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {e}")

    return StreamingResponse(buf, media_type="image/png")
