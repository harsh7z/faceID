import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from siamese_network import SiameseNetwork

IMG_SIZE = 105
THRESHOLD = 0.8  # Tune this based on training

# Load model
model = SiameseNetwork()
model.load_state_dict(torch.load("models/siamese_model.pt"))
model.eval()

# Preprocess face
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_pil = Image.fromarray(face_img)
    return transform(face_pil).unsqueeze(0)

def get_face_embedding(face_tensor):
    with torch.no_grad():
        return model.forward_once(face_tensor)

def verify_user():
    cap = cv2.VideoCapture(1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("Showing webcam feed. Press 'q' to verify a face.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Press 'q' to verify", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) == 0:
        print("No face detected.")
        return

    # Take first detected face
    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    input_face_tensor = preprocess_face(face_img)
    input_embedding = get_face_embedding(input_face_tensor)

    best_score = float("inf")
    best_user = "Unknown"

    for user in os.listdir("data/users"):
        user_dir = os.path.join("data/users", user)
        if not os.path.isdir(user_dir):
            continue

        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            ref_img = Image.open(img_path).convert("L")
            ref_tensor = transform(ref_img).unsqueeze(0)
            ref_embedding = get_face_embedding(ref_tensor)

            dist = torch.nn.functional.pairwise_distance(input_embedding, ref_embedding)
            score = dist.item()

            if score < best_score:
                best_score = score
                best_user = user

    print(f"🔍 Best match: {best_user} (score: {best_score:.4f})")
    if best_score < THRESHOLD:
        print(f"Verified as: {best_user}")
    else:
        print("Face not recognized.")

if __name__ == "__main__":
    verify_user()
