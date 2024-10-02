
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn #to execute - insead of console cmd
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
#better way to test is postman,/docs
app = FastAPI() #instance

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
# MODEL = tf.keras.models.load_model("C:\RDR2\POTATO\models\potatoes.h5")
MODEL = tf.keras.models.load_model(r"C:\RDR2\POTATO\models\potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Mini-project"
#async and  await serve requests one by one by keeping in suspend mode

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict") # to create data
async def predict(
        file: UploadFile = File(...) #file as i/p & UploadFile is dtype
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0) # It reads multiple images one by one
    #expand_dims : add 1 extra dimension
    # 0 is an axis for extra dimension
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) #in batch 0th img
    #confidence is max value in softmax
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
