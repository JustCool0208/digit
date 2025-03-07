from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import base64
import io
from PIL import Image

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

class ImageData(BaseModel):
    image: str
    operation: str
    second_number: int

def preprocess_image(base64_string):
    # Convert base64 to tensor (grayscale 28x28)
    image_data = base64.b64decode(base64_string.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)

@app.post("/predict")
def predict_digit(data: ImageData):
    try:
        processed_image = preprocess_image(data.image)
        with torch.no_grad():
            output = model(processed_image)
            first_number = int(torch.argmax(output))

        second_number = data.second_number
        operation = data.operation
        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide" and second_number != 0:
            result = first_number / second_number
        else:
            raise HTTPException(status_code=400, detail="Invalid operation or division by zero")
        

        return {"first_digit": first_number, "second_digit": second_number, "operation": operation, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
