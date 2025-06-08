from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import base64
import numpy as np
import cv2
from src.car_blur import find_car_on_image_and_blur

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")
uploaded_files = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            content={"error": "File is not an image"},
            status_code=400
        )

    content = await file.read()
    file_size = len(content)

    uploaded_files[file.filename] = {
        "content_type": file.content_type,
        "content": content,
        "file_size": file_size
    }

    return RedirectResponse(
        url=f"/image-info?filename={file.filename}",
        status_code=303
    )


@app.get("/image-info", response_class=HTMLResponse)
async def image_info(request: Request, filename: str):
    file_info = uploaded_files.get(filename)

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
    result = find_car_on_image_and_blur(img)['blurred_image']

    # Кодируем результат обратно в JPEG
    _, buffer = cv2.imencode('.jpg', result)  # можно '.png' если хочешь PNG

    # Получаем байты
    result_bytes = buffer.tobytes()

    # Кодируем в base64
    base64_result = base64.b64encode(result_bytes).decode('utf-8')

    # Готовим data URI
    result_data_uri = f"data:image/jpeg;base64,{base64_result}"

    return templates.TemplateResponse("info.html", {
        "request": request,
        "filename": filename,
        "content_type": file_info["content_type"],
        "file_size": file_info["file_size"],
        "image": data_uri,
        "processed_image": result_data_uri
    })