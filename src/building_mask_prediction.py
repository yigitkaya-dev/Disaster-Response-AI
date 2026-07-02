"""
UNetFormer Localization Prediction Script
Loads trained UNetFormer weights in a local environment
to process individual pre-disaster satellite images and output 
binary structural maps.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# =====================================================================
# 1. INITIALIZATION of MODEL ARCHITECTURE (UNetFormer with ResNet-34 Encoder)
# =====================================================================

class EfficientGlobalLocalAttention(nn.Module):
    """
    Combines localized depthwise convolutions with a spatially-reduced
    global matrix attention mechanism. Prevents quadratic memory explosion on GPUs.
    """
    def __init__(self, channels):
        super(EfficientGlobalLocalAttention, self).__init__()
        # Local Attention Pathway
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # Global Attention Pathway (Downsampled to 4x4 for memory safety)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc_query = nn.Conv2d(channels, channels, kernel_size=1)
        self.fc_key   = nn.Conv2d(channels, channels, kernel_size=1)
        self.fc_value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax  = nn.Softmax(dim=-1)
        self.proj     = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_branch(x)
        # Linearized global attention computation
        q = self.fc_query(x).flatten(2) 
        pooled_x = self.pool(x)
        k = self.fc_key(pooled_x).flatten(2)   
        v = self.fc_value(pooled_x).flatten(2) 
        energy = torch.bmm(q.transpose(1, 2), k) 
        attention = self.softmax(energy)
        global_feat = torch.bmm(v, attention.transpose(1, 2)).view(B, C, H, W)
        return self.proj(local_feat + global_feat)


class UNetFormerDecoderBlock(nn.Module):
    """Upsamples feature maps and fuses them with high-resolution encoder skip connections."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetFormerDecoderBlock, self).__init__()
        self.attention = EfficientGlobalLocalAttention(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x = self.attention(x)
        x = self.upsample(x)
        x = torch.cat([x, skip_features], dim=1)
        return self.conv(x)


class UNetFormer(nn.Module):
    """Full UNetFormer model integrating ResNet-34 encoder and EGLA decoders."""
    def __init__(self, num_classes=1):
        super(UNetFormer, self).__init__()
        resnet = models.resnet34(pretrained=False) # Local file overrides optimization download
        # Encoder
        self.encoder_initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) 
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
        # Decoder
        self.decoder3 = UNetFormerDecoderBlock(512, 256, 256)
        self.decoder2 = UNetFormerDecoderBlock(256, 128, 128)
        self.decoder1 = UNetFormerDecoderBlock(128, 64, 64)
        # Head Projection
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x_init = self.encoder_initial(x) 
        s1 = self.layer1(x_init)   
        s2 = self.layer2(s1)       
        s3 = self.layer3(s2)       
        bottleneck = self.layer4(s3) 
        d3 = self.decoder3(bottleneck, s3) 
        d2 = self.decoder2(d3, s2)         
        d1 = self.decoder1(d2, s1)         
        return self.final_conv(self.final_upsample(d1))


# =====================================================================
# 2. From Mask to Polygon
# =====================================================================
'''
def extract_building_polygons(mask_visual):
    """
    Converts a binary pixel mask into a list of geometric building boundary bounding boxes.
    
    Translates raster pixel grids into vector coordinates. For each isolated white 
    blob (building), it extracts the bounding perimeter, allowing seamless integration 
    with polygon cropping routines.
        
    Returns:
        A list of dictionaries, where each dict contains the coordinates 
        {'minx': val, 'miny': val, 'maxx': val, 'maxy': val}
    """
    # 1. Find continuous white boundaries in the binary mask
    # cv2.RETR_EXTERNAL only looks at the outside boundary of the building.
    contours, _ = cv2.findContours(mask_visual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    building_bounds = []
    
    for contour in contours:
        # Ignore tiny artifacts or pixel noise (e.g., individual bright pixels smaller than 10 total area)
        #if cv2.contourArea(contour) < 10:
        #    continue
            
        # Extract standard bounding box coordinates using OpenCV
        # x, y are the top-left starting pixels; w, h are the width and height of the box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Map to coordinates
        minx = x
        miny = y
        maxx = x + w
        maxy = y + h
        
        building_bounds.append({
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy
        })
        
    return building_bounds
'''

def extract_building_polygons(mask_visual):
    """
    Applies Distance Transform and Watershed segmentation to break merged building 
    clusters into individual, cleanly bounded structures.
    """
    # Ensure mask is strictly binary (0 or 255)
    _, binary = cv2.threshold(mask_visual, 127, 255, cv2.THRESH_BINARY)
    
    # 1. Compute the Distance Transform
    # Calculates the Euclidean distance from every white pixel to the nearest black background pixel
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # 2. Isolate the "peaks" (the independent centers of each house)
    # Adjust the multiplier (0.35 - 0.5) if you want to be more or less aggressive at splitting
    _, sure_fg = cv2.threshold(dist_transform, 0.32 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Create structural markers for the Watershed algorithm
    # Unknown region represents the transition areas/boundaries where houses blend
    unknown = cv2.subtract(binary, sure_fg)
    
    # Label independent foreground objects with distinct numbers (0 is background)
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that sure background is 1 instead of 0
    markers = markers + 1
    # Mark the unknown boundary regions as 0 for the watershed algorithm to solve
    markers[unknown == 255] = 0
    
    # 4. Apply Watershed (Requires a 3-channel image array, so we convert our binary mask format)
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_mask, markers)
    
    # Watershed modifies the markers matrix: boundaries between separated objects are marked with -1
    
    # 5. Extract independent bounding boxes from the updated markers
    building_bounds = []
    
    # Loop through unique object IDs (skipping 1 which is background, and -1 which are boundaries)
    unique_markers = np.unique(markers)
    for marker_id in unique_markers:
        if marker_id <= 1:
            continue
            
        # Create a temporary binary mask for just this single isolated building instance
        building_mask = np.uint8(markers == marker_id)
        
        # Find the contour boundary of this specific split object
        contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Grab the largest contour fragment for safety
            contour = max(contours, key=cv2.contourArea)
            
            # Filter out tiny residual pixel noise anomalies
            if cv2.contourArea(contour) < 15:
                continue
                
            # Extract standard bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            building_bounds.append({
                'minx': x,
                'miny': y,
                'maxx': x + w,
                'maxy': y + h
            })
            
    return building_bounds

# =====================================================================
# 3. Prediction
# =====================================================================

def predict_single_image(image_path, weight_path, output_path):
    """
    Loads local weights, preprocesses a post-disaster image, 
    and saves the predicted building mask.
    """
    # Auto-fallback processing environment config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize network model structure
    model = UNetFormer(num_classes=1)
    
    # Load parameters safely across varying platforms 
    checkpoint = torch.load(weight_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # Read image and preserve spatial proportions
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Invalid path: {image_path}")
        
    h, w, _ = raw_image.shape
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Standardized ImageNet scale-invariant pre-processing
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor_input = transform(image=image_rgb)['image'].unsqueeze(0).to(device)
    
    # Forward Pass Evaluation
    #print("⏳ Evaluating segmentation maps...")
    with torch.no_grad():
        logits = model(tensor_input)
        # Apply sigmoid classification boundary threshold at 0.5
        prediction = (torch.sigmoid(logits) > 0.5).float()
        
    # Translate processing matrix variables back into byte visuals
    mask_np = prediction.squeeze().cpu().numpy()
    mask_visual = (mask_np * 255).astype(np.uint8)
    
    # Dynamic catch-all resize to safeguard spatial dimension integrity
    if mask_visual.shape != (h, w):
        mask_visual = cv2.resize(mask_visual, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # Write mask image file out to local drive folder
    cv2.imwrite(output_path, mask_visual)
    print(f"Success! Localization mask extracted cleanly to: {output_path}")

    # Extract the polygon coordinates
    building_coordinates = extract_building_polygons(mask_visual)
    print(f"Extracted {len(building_coordinates)} distinct buildings from this satellite frame.")
    
    # Return both the mask array and the coordinate list
    return mask_visual, building_coordinates

def visualize_coordinate_verification(image_path, building_bounds):
    """
    Overlays the extracted coordinate boxes directly onto the original image 
    to visually test and confirm bounding accuracy.
    """
    # 1. Load the original satellite image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Verification Error: Could not read image at {image_path}")
        return
        
    # Make a copy so we don't mutate or ruin the original cached image array
    overlay_img = img.copy()
    
    # 2. Iterate through each building's bounding dictionary
    for i, box in enumerate(building_bounds):
        # Extract the coordinate properties
        minx = box['minx']
        miny = box['miny']
        maxx = box['maxx']
        maxy = box['maxy']
        
        # 3. Draw a bounding rectangle over the building
        # Parameters: (image, top-left coordinate, bottom-right coordinate, color, thickness)
        # Color is BGR. (0, 255, 0) gives a vibrant, high-contrast neon green box.
        cv2.rectangle(overlay_img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        
        # Label each box with its index number for discrete identification
        cv2.putText(overlay_img, str(i), (minx, miny - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 4. Convert back to RGB for proper color display in Matplotlib
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    
    # 5. Print/Display the final composite image
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_rgb)
    plt.title(f"Coordinate Verification Map — {len(building_bounds)} Green Bounding Enclosures Verified")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # --- Local Parameter Definitions ---
    # Update these paths based on your local directory structure
    LOCAL_WEIGHTS = "src/unetformer_best_localization.pth"
    IMAGE    = "C:/Users/nicol/OneDrive/Documents/FAU/Summer 2026/Grad Project/Disaster-Response-AI/Dataset/tier3/images/joplin-tornado_00000000_pre_disaster.png"
    OUTPUT_MASK   = "output/predicted_localization_mask.png"
    
    # Ensure local directory paths exist
    os.makedirs("output", exist_ok=True)
    
    # Run the local inference pipeline
    mask_visual, building_coordinates = predict_single_image(IMAGE, LOCAL_WEIGHTS, OUTPUT_MASK)
    #print(build_list)

    visualize_coordinate_verification(IMAGE, building_coordinates)