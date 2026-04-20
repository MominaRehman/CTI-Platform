#!/usr/bin/env python
"""
Training Script for Deep Learning Models
Text GNN (Graph Neural Network) and Image ViT (Vision Transformer)
"""

import requests
import time
import os
import sys

API_URL = "http://127.0.0.1:8000/api/training"

print("="*70)
print("🤖 Deep Learning Model Training")
print("   Text GNN (Graph Neural Network)")
print("   Image ViT (Vision Transformer)")
print("="*70)

# Check if server is running
try:
    response = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print("✅ Backend server is running")
except:
    print("❌ Backend server is not running!")
    print("Please start the backend first: python run.py")
    sys.exit(1)

print("\n" + "-"*70)
print("Step 1: Upload Text Dataset for GNN Training")
print("-"*70)

# Check for text dataset in uploads folder
text_paths = [
    r"D:\cyber threat intel\backend\uploads\text\corrected_text_data.csv",
    r"D:\cyber threat intel\corrected_text_data.csv",
]

text_found = False
for csv_path in text_paths:
    if os.path.exists(csv_path):
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/upload-text", files=files)
            print(f"✅ Uploaded text dataset from: {csv_path}")
            print(f"   Response: {response.json()}")
            text_found = True
            break

if not text_found:
    print(f"⚠️ Text CSV file not found!")
    print("Creating sample text dataset...")
    sample_path = "uploads/text/sample_text_data.csv"
    with open(sample_path, 'w') as f:
        f.write("text,label\n")
        f.write("Looking for 9mm ammunition,gun\n")
        f.write("Pure cocaine for sale,drug\n")
        f.write("Cyanide extraction method,poison\n")
        f.write("AR-15 lower receiver,gun\n")
        f.write("High quality MDMA crystals,drug\n")
        f.write("Arsenic powder available,poison\n")
    
    with open(sample_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/upload-text", files=files)
        print(f"✅ Sample dataset created and uploaded")

print("\n" + "-"*70)
print("Step 2: Upload Image Dataset for ViT Training")
print("-"*70)

# Check for images in uploads/images folder
image_base = r"D:\cyber threat intel\backend\uploads\images"
categories = ['drugs', 'firearms', 'poison']

if os.path.exists(image_base):
    print(f"✅ Found image folder: {image_base}")
    for category in categories:
        category_path = os.path.join(image_base, category)
        if os.path.exists(category_path):
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if image_files:
                print(f"\n📂 Uploading {len(image_files)} images to {category}...")
                files_to_upload = []
                for img_file in image_files[:20]:  # Limit to 20 images per category
                    img_path = os.path.join(category_path, img_file)
                    files_to_upload.append(('files', (img_file, open(img_path, 'rb'), 'image/jpeg')))
                
                if files_to_upload:
                    response = requests.post(f"{API_URL}/upload-images/{category}", files=files_to_upload)
                    print(f"✅ Uploaded {len(files_to_upload)} images to {category}")
                    
                    # Close all file handles
                    for _, (name, f, _) in files_to_upload:
                        f.close()
            else:
                print(f"⚠️ No images found in {category_path}")
        else:
            print(f"⚠️ Category folder not found: {category_path}")
else:
    print(f"⚠️ Image folder not found at {image_base}")
    print("Creating sample images...")
    
    # Create sample image folders and images
    for category in categories:
        category_path = os.path.join(image_base, category)
        os.makedirs(category_path, exist_ok=True)
        
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        for i in range(3):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(category_path, f"sample_{i+1}.jpg")
            img.save(img_path)
            print(f"   Created sample image: {img_path}")
    
    # Now upload the sample images
    for category in categories:
        category_path = os.path.join(image_base, category)
        files_to_upload = []
        for img_file in os.listdir(category_path)[:5]:
            img_path = os.path.join(category_path, img_file)
            files_to_upload.append(('files', (img_file, open(img_path, 'rb'), 'image/jpeg')))
        
        if files_to_upload:
            response = requests.post(f"{API_URL}/upload-images/{category}", files=files_to_upload)
            print(f"✅ Uploaded sample images to {category}")
            for _, (name, f, _) in files_to_upload:
                f.close()

print("\n" + "-"*70)
print("Step 3: Train Text GNN (Graph Neural Network) - 50 epochs")
print("-"*70)

train_data = {
    "model_type": "text_gnn",
    "data_path": "uploads/text/corrected_text_data.csv",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}
print("🚀 Starting Text GNN training...")
response = requests.post(f"{API_URL}/start", json=train_data)
if response.status_code == 200:
    result = response.json()
    print(f"✅ Text GNN training complete!")
    print(f"   Architecture: Graph Neural Network")
    print(f"   Final Accuracy: {result.get('final_accuracy', 0):.2%}")
else:
    print(f"❌ Training failed: {response.text}")

print("\n" + "-"*70)
print("Step 4: Train Image ViT (Vision Transformer) - 10 epochs")
print("-"*70)

train_data = {
    "model_type": "image_vit",
    "data_path": "uploads/images",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0001
}
print("🚀 Starting Image ViT training...")
response = requests.post(f"{API_URL}/start", json=train_data)
if response.status_code == 200:
    result = response.json()
    print(f"✅ Image ViT training complete!")
    print(f"   Architecture: Vision Transformer")
    print(f"   Final Accuracy: {result.get('final_accuracy', 0):.2%}")
else:
    print(f"❌ Training failed: {response.text}")

print("\n" + "-"*70)
print("Step 5: Check Deep Learning Model Status")
print("-"*70)

response = requests.get(f"{API_URL}/status")
if response.status_code == 200:
    status = response.json()
    print(f"📊 Text GNN (GNN): {'✅ Trained' if status['text_gnn']['trained'] else '❌ Not trained'}")
    print(f"📊 Image ViT (ViT): {'✅ Trained' if status['image_vit']['trained'] else '❌ Not trained'}")

print("\n" + "="*70)
print("🎉 Deep Learning Training Complete!")
print("="*70)
print("\n💡 Next Steps:")
print("1. Restart the backend to load trained models")
print("2. Test Text GNN: POST /api/predict/text")
print("3. Test Image ViT: POST /api/predict/image")
print("4. Scrape onion sites: POST /api/onion/scrape")