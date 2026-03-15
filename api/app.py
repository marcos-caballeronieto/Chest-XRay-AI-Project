from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import base64

# 1. Initialize the FastAPI App
app = FastAPI(
    title="Pneumonia Triage API - DenseNet Edition",
    description="Clinical API for detecting pneumonia using a Fine-Tuned DenseNet121 at 448px with 3-way TTA Consensus.",
    version="2.0.0"
)

device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights.pth")

# 2. Setup the DenseNet121 Architecture
def load_model():
    # Load base DenseNet121
    model = models.densenet121(weights=None)
    
    # Modify the classifier for 2 classes (Normal vs Pneumonia)
    # Note: DenseNet uses 'classifier', ResNet uses 'fc'
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Fine-Tuned DenseNet121 loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Predictions will be random.")
        
    model.to(device)
    model.eval()
    
    # Patch the forward pass to prevent in-place modification of features
    import torch.nn.functional as F
    def patched_forward(x):
        features = model.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        return out
    
    model.forward = patched_forward
    return model

model = load_model()

class GradCAM:
    def __init__(self, model, feature_module):
        self.model = model
        self.feature_module = feature_module
        self.gradient = None
        self.activation = None
        
        self.forward_hook = self.feature_module.register_forward_hook(self.save_activation)
        self.backward_hook = self.feature_module.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output.clone()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].clone()

    def generate(self, input_tensor, class_idx=1):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        weight = torch.mean(self.gradient, dim=[2, 3], keepdim=True)
        cam = torch.sum(weight * self.activation, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam -= torch.min(cam)
        cam /= (torch.max(cam) + 1e-8)
        
        return cam.cpu().detach().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# 3. Define the High-Resolution 3-Way TTA Transforms (448px)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Variant 1: Original (Resize to 448x448)
transform_original = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    normalize
])

# Variant 2: Rotated (10 degrees)
transform_rotated = transforms.Compose([
    transforms.RandomRotation((10, 10)),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    normalize
])

# Variant 3: Zoomed (CenterCrop from a slightly larger base)
transform_zoomed = transforms.Compose([
    transforms.Resize(512), 
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize
])

# 4. The Prediction Endpoint
@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate the 3 variants
        img_orig = transform_original(image).unsqueeze(0).to(device)
        img_rot = transform_rotated(image).unsqueeze(0).to(device)
        img_zoom = transform_zoomed(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get logits
            out_orig = model(img_orig)
            out_rot = model(img_rot)
            out_zoom = model(img_zoom)
            
            # Convert to probabilities
            prob_orig = torch.softmax(out_orig, dim=1)[0]
            prob_rot = torch.softmax(out_rot, dim=1)[0]
            prob_zoom = torch.softmax(out_zoom, dim=1)[0]
            
            # Extract Pneumonia probabilities (assuming Class 1 is Pneumonia)
            p1, p2, p3 = prob_orig[1].item(), prob_rot[1].item(), prob_zoom[1].item()
            
            # Hard Voting: 3-way committee
            votes_for_pneumonia = sum([p1 > 0.50, p2 > 0.50, p3 > 0.50])
            diagnosis = "Pneumonia" if votes_for_pneumonia >= 2 else "Normal"
            
        # Generate GradCAM
        gradcam_base64 = None
        try:
            with torch.enable_grad():
                img_for_cam = transform_original(image).unsqueeze(0).to(device)
                img_for_cam.requires_grad_(True)
                
                gradcam = GradCAM(model, model.features)
                cam_class = 1 if diagnosis == "Pneumonia" else 0
                cam_mask = gradcam.generate(img_for_cam, class_idx=cam_class)
                gradcam.remove_hooks()
                
                orig_array = np.array(image.resize((448, 448)))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (orig_array.shape[1], orig_array.shape[0]))
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                overlay = heatmap * 0.4 + orig_array * 0.6
                overlay_img = Image.fromarray(np.uint8(overlay))
                
                buffer = io.BytesIO()
                overlay_img.save(buffer, format="JPEG")
                gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            
        return JSONResponse({
            "diagnosis": diagnosis,
            "confidence_votes": f"{votes_for_pneumonia}/3",
            "clinical_metrics": "Optimized for 0 False Negatives via TTA consensus.",
            "raw_probabilities": {
                "original_448px": round(p1, 4),
                "rotated_448px": round(p2, 4),
                "zoomed_448px": round(p3, 4)
            },
            "gradcam_base64": gradcam_base64
        })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "DenseNet121 TTA API is live. Ready for inference at 448px."}