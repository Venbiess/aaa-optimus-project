from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import base64
import numpy as np
import cv2
import time
import hashlib
from torch.cuda import is_available as cuda_is_available
from src.config import HASH_SIZE
from src.car_blur import find_car_on_image_and_blur


device = 'cuda' if cuda_is_available() else 'cpu'
app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static"), name="static")

templates = Jinja2Templates(directory="src/templates")
uploaded_files = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def generate_file_hash(filename: str, file_size: int) -> str:
    timestamp = str(time.time())
    data = f"{timestamp}-{filename}-{file_size}"
    hash_digest = hashlib.sha256(data.encode()).hexdigest()
    return hash_digest[:HASH_SIZE]


@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form(None),
    format: str = Form(None),
    blur_level: int = Form(47)
):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            content={"error": "File is not an image"},
            status_code=400
        )

    content = await file.read()

    unique_id = generate_file_hash(file.filename, len(content))
    uploaded_files[unique_id] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "content": content,
        "file_size": len(content)
    }

    return RedirectResponse(
        url=f"/image-info?image_id={unique_id}&model={model}&format={format}&blur_level={blur_level}",
        status_code=303
    )


@app.get("/image-info", response_class=HTMLResponse)
async def image_info(
    request: Request,
    image_id: str,
    model: str,
    format: str,
    blur_level: int
):
    time_start = time.time()
    file_info = uploaded_files.get(image_id)

    if not file_info:
        return HTMLResponse(
            content="<h1>File not found!</h1>",
            status_code=404
        )

    # Формируем Data URI
    base64_data = base64.b64encode(file_info["content"]).decode("utf-8")
    data_uri = f"data:{file_info['content_type']};base64,{base64_data}"

    image_bytes = file_info["content"]
    bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
    result = find_car_on_image_and_blur(img, kernel_size=blur_level)['blurred_image']

    # Кодируем результат обратно в JPEG
    _, buffer = cv2.imencode('.jpg', result)
    result_bytes = buffer.tobytes()
    base64_result = base64.b64encode(result_bytes).decode('utf-8')
    result_data_uri = f"data:image/jpeg;base64,{base64_result}"

    return templates.TemplateResponse("info.html", {
        "request": request,
        "image_id": image_id,
        "content_type": file_info["content_type"],
        "file_size": file_info["file_size"],
        "image": data_uri,
        "processed_image": result_data_uri,
        "model": model,
        "format": format,
        "blur_level": blur_level,
        "time_spent": f'{round(time.time() - time_start, 2)} (сек.)',
        "accelerator": device
    })