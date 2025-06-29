import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import joblib
from pydantic import BaseModel
import json
import os
from pathlib import Path

# Загрузка модели
try:
    model = joblib.load("knn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

app = FastAPI(title="Prediction HA_API")

# Создаем директорию для сохранения результатов
RESULTS_DIR = "results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

class PredictionResult(BaseModel):
    filename: str
    predictions: list
    model_name: str

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение CSV файла
        df = pd.read_csv(file.file)
        
        if df.empty:
            return JSONResponse(
                status_code=400,
                content={"message": "Файл пуст или не может быть прочитан"}
            )
        
        # Предсказание
        predictions = model.predict(df)
        
        # Подготовка результатов
        result = {
            "filename": file.filename,
            "predictions": predictions.tolist(),
            "model_name": str('kNN Model')
        }
        
        # Сохранение в JSON
        json_filename = f"{Path(file.filename).stem}_predictions.json"
        json_path = os.path.join(RESULTS_DIR, json_filename)
        with open(json_path, 'w') as f:
            json.dump(result, f)
        
        # Генерация HTML отчета
        html_filename = f"{Path(file.filename).stem}_predictions.html"
        html_path = os.path.join(RESULTS_DIR, html_filename)
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Результаты предсказания</h1>
                    <p><strong>Файл:</strong> {file.filename}</p>
                    <p><strong>Модель:</strong> {model.__class__.__name__}</p>
                    
                    <h2>Предсказания</h2>
                    <table>
                        <tr><th>ID</th><th>Prediction</th></tr>
                        {''.join(f'<tr><td>{i}</td><td>{pred}</td></tr>' 
                                for i, pred in enumerate(result['predictions']))}
                    </table>
                </div>
            </body>
            </html>
            """)
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Ошибка обработки файла: {str(e)}"}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(
        status_code=404,
        content={"message": "Файл не найден"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)