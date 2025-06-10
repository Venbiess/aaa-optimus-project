from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import base64
from PIL import Image
import numpy as np
import io
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


def mask_to_data_uri(mask_array: np.ndarray):
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    base64_mask = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_mask}"


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

    if ("mask" not in file_info) or (model != file_info["model"]):
        image_bytes = file_info["content"]
        bytes = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
        result = find_car_on_image_and_blur(img, kernel_size=blur_level, model=model)

        # Кодируем результат обратно в JPEG
        _, buffer = cv2.imencode('.jpg', result['blurred_image'])
        result_bytes = buffer.tobytes()
        base64_result = base64.b64encode(result_bytes).decode('utf-8')
        result_data_uri = f"data:image/jpeg;base64,{base64_result}"

        uploaded_files[image_id]["mask"] = mask_to_data_uri(result['mask'])
        uploaded_files[image_id]["data_uri"] = result_data_uri
        uploaded_files[image_id]["time_spent"] = round(time.time() - time_start, 2)
        uploaded_files[image_id]["model"] = model

    return templates.TemplateResponse("info.html", {
        "request": request,
        "image_id": image_id,
        "content_type": file_info["content_type"],
        "file_size": file_info["file_size"],
        "image": data_uri,
        "mask": file_info["mask"],
        "processed_image": file_info["data_uri"],
        "model": model,
        "format": format,
        "blur_level": blur_level,
        "time_spent": f'{file_info["time_spent"]} (сек.)',
        "accelerator": device
    })
