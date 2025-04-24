import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
import cv2
from skimage.measure import label, regionprops

# Constants
DAMAGE_CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
DAMAGE_COLORS = {
    "no-damage": (0, 255, 0, 50),
    "minor-damage": (0, 0, 255, 50),
    "major-damage": (255, 69, 0, 50),
    "destroyed": (255, 0, 0, 50),
    "un-classified": (255, 255, 255, 50)
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
@st.cache_resource
def load_models():
    # Load segmentation model
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.classifier[-1] = nn.Conv2d(256, len(DAMAGE_CLASSES), kernel_size=(1, 1), stride=(1, 1))
    seg_model.load_state_dict(torch.load('segmentation_model.pth', map_location=DEVICE)['model_state_dict'])
    seg_model.to(DEVICE)
    seg_model.eval()
    
    # Load classification model
    cls_model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    cls_model.classifier[1] = nn.Linear(cls_model.classifier[1].in_features, len(DAMAGE_CLASSES))
    cls_model.load_state_dict(torch.load('crops/efficientnetb0_damage_best.pth', map_location=DEVICE))
    cls_model.to(DEVICE)
    cls_model.eval()
    
    return seg_model, cls_model

# Image transforms
seg_tfm = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cls_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_colored_mask(pred_mask):
    """Convert prediction mask to colored mask using DAMAGE_COLORS"""
    colored_mask = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
    for i, color in enumerate(DAMAGE_COLORS.values()):
        colored_mask[pred_mask == i] = color
    return colored_mask

def get_bounding_boxes(mask):
    """Get bounding boxes for each connected component in the mask"""
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8)
    # Label connected components
    labeled_mask = label(binary_mask)
    # Get region properties
    regions = regionprops(labeled_mask)
    return regions

def crop_and_classify(image, mask, regions, cls_model):
    """Crop regions from image and classify them"""
    results = []
    image_np = np.array(image)
    
    for region in regions:
        # Get bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        
        # Add some padding
        padding = 10
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(image_np.shape[0], max_row + padding)
        max_col = min(image_np.shape[1], max_col + padding)
        
        # Crop the region
        crop = image_np[min_row:max_row, min_col:max_col]
        crop_pil = Image.fromarray(crop)
        
        # Classify the crop
        with torch.no_grad():
            crop_tensor = cls_tfm(crop_pil).unsqueeze(0).to(DEVICE)
            cls_pred = cls_model(crop_tensor)
            cls_prob = torch.softmax(cls_pred, dim=1)[0]
            cls_label = DAMAGE_CLASSES[cls_pred.argmax().item()]
        
        results.append({
            'crop': crop_pil,
            'bbox': (min_col, min_row, max_col, max_row),
            'class': cls_label,
            'probabilities': cls_prob.cpu().numpy()
        })
    
    return results

# Load models
seg_model, cls_model = load_models()

# Streamlit app
st.title('Building Damage Assessment')

uploaded_file = st.file_uploader("Upload a building image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Get predictions
    with torch.no_grad():
        # Segmentation
        img_tensor = seg_tfm(image).unsqueeze(0).to(DEVICE)
        seg_output = seg_model(img_tensor)['out']
        seg_pred = seg_output.argmax(dim=1).squeeze().cpu().numpy()
        colored_mask = create_colored_mask(seg_pred)
        
        # Get regions and classify them
        regions = get_bounding_boxes(seg_pred)
        results = crop_and_classify(image, seg_pred, regions, cls_model)
    
    # Display results
    st.subheader('Segmentation Mask')
    st.image(colored_mask, use_container_width=True)
    
    st.subheader('Detected Buildings')
    for i, result in enumerate(results):
        col1, col2 = st.columns(2)
        with col1:
            st.image(result['crop'], caption=f'Building {i+1}')
        with col2:
            st.write(f'Predicted class: {result["class"]}')
