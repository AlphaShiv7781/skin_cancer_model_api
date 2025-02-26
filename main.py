import os
from pyexpat import model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import uvicorn
import pydicom

app = FastAPI()

# Load TFLite Model
MODEL_PATH = "models/skin_cancer_model.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")






def preprocess_image(image: Image.Image):
    """Preprocess image to match model input"""
    image = image.convert('RGB')  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize to model input shape
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array




@app.get("/")
def home():
    return {"message": "InstaScan API is running!"}

def process_imagePneumonia(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predictPneumonia")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    
    if filename.endswith(".dcm"):
        dicom_img = pydicom.dcmread(io.BytesIO(await file.read()))
        img_array = dicom_img.pixel_array
        img = Image.fromarray(img_array).convert("RGB")
    else:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    image = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0] * 100
    return {"disease-confidence": f"{prediction:.2f}%"}



# Skin Cancer
@app.post("/predictSkinCancer")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_data = preprocess_image(image)

        # Ensure correct input shape
        if input_data.shape != tuple(input_details[0]['shape']):
            raise HTTPException(status_code=400, detail="Invalid input shape")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        pred_dict = dict(zip(class_names, predictions.tolist()))
        disease_confidence = (1 - pred_dict["nv"]) * 100  # Convert to percentage

        return {"disease_confidence": f"{disease_confidence:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 for local testing
    uvicorn.run(app, host="0.0.0.0", port=port)
