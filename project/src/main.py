from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import base64

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

    return templates.TemplateResponse("info.html", {
        "request": request,
        "filename": filename,
        "content_type": file_info["content_type"],
        "file_size": file_info["file_size"],
        "image": data_uri
    })