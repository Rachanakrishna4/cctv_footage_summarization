import cv2, os, glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Load ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def extract_frames(video_path, output_folder="frames", interval=30):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved}.jpg", frame)
            saved += 1
        frame_count += 1
    cap.release()

def get_features(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(tensor)
    return features.numpy()

def cluster_frames(frame_folder="frames", summary_folder="summary_frames", k=10):
    os.makedirs(summary_folder, exist_ok=True)
    features, paths = [], glob.glob(f"{frame_folder}/*.jpg")
    
    # Sort paths by frame number to preserve order
    paths = sorted(paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    for p in paths:
        features.append(get_features(p))
    features = np.vstack(features)
    
    actual_k = min(k, len(features))
    if actual_k < 1:
        print("Not enough frames to summarize.")
        return

    kmeans = KMeans(n_clusters=actual_k).fit(features)
    centers = kmeans.cluster_centers_
    chosen = []
    for i in range(actual_k):
        idx = np.argmin(np.linalg.norm(features - centers[i], axis=1))
        chosen.append(paths[idx])
    
    # ✅ Sort chosen frames by their original index
    chosen = sorted(chosen, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    for i, c in enumerate(chosen):
        img = cv2.imread(c)
        cv2.imwrite(f"{summary_folder}/keyframe_{i}.jpg", img)

def build_video(frame_folder="summary_frames", output="static/summary_output.mp4", fps=1/2):
    frames = sorted(glob.glob(f"{frame_folder}/*.jpg"))
    if not frames: return
    first_frame = cv2.imread(frames[0])
    h, w, _ = first_frame.shape
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for f in frames:
        img = cv2.imread(f)
        out.write(img)
    out.release()