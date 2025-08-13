import cv2
import numpy as np
import psycopg2
import os
from PIL import Image
from imgbeddings import imgbeddings  # Import the embedding module
from dotenv import load_dotenv

load_dotenv()

AIVEN_PASSWORD = os.getenv("AIVEN_PASSWORD")


HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
STORED_FACES_DIR = "stored_faces"
DB_CONFIG = {
    "dbname": "defaultdb",
    "user": "avnadmin",
    "password": AIVEN_PASSWORD,
    "host": "pg-3993ef27-sheikh-first.f.aivencloud.com",
    "port": 24222,
    "sslmode": "require",
}

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
os.makedirs(STORED_FACES_DIR, exist_ok=True)

def detect_faces():
    """Detect faces in an image and save them as separate files."""
    file_name = "Harry_potter_students.jpg"
    img = cv2.imread(file_name)

    if img is None:
        print(f"Error: {file_name} not found.")
        return []

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    face_files = []
    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = img[y: y + h, x: x + w]
        
        resized_face = cv2.resize(cropped_face, (200, 200)) 
        
        face_path = os.path.join(STORED_FACES_DIR, f"{i}.jpg")
        cv2.imwrite(face_path, resized_face) 
        face_files.append(face_path)

    return face_files

def save_faces_to_db(face_files):
    """Save faces to the database along with their embeddings."""
    # Initialize the embedding model
    ibed = imgbeddings()

    conn = psycopg2.connect(
        dbname="defaultdb",
        user="avnadmin",
        password=AIVEN_PASSWORD,
        host="pg-3993ef27-sheikh-first.f.aivencloud.com",
        port=24222,
        sslmode="require"
    )
    cur = conn.cursor()

    for filename in face_files:
        with open(filename, "rb") as f:
            face_image = f.read()  # Read image as binary data

        # Get the embedding for the face image
        img = Image.open(filename).convert("RGB")
        embedding = ibed.to_embeddings(img)[0].tolist()  # Get the embedding as a list

        # Insert the image and embedding into the database
        cur.execute(
            "INSERT INTO pictures (picture, embedding) VALUES (%s, %s);",
            (psycopg2.Binary(face_image), embedding)  # Store the embedding as a list of floats
        )

    conn.commit()
    cur.close()
    conn.close()

def main():
    face_files = detect_faces()
    if face_files:
        save_faces_to_db(face_files)
    else:
        print("No faces detected.")

if __name__ == "__main__":
    main()
