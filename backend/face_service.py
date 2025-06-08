import cv2
import numpy as np
import base64
import io
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize face detection and recognition models
embedder = FaceNet()
detector = MTCNN()

def base64_to_opencv_image(base64_string):
    """Convert base64 string to OpenCV image"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    image_bytes = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(image_bytes))
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return opencv_image

def extract_face_embedding(image):
    """Extract face embedding from image using MTCNN and FaceNet"""
    faces = detector.detect_faces(image)

    if not faces:
        return None

    largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = largest_face['box']

    cropped_face = image[y:y+h, x:x+w]
    resized_face = cv2.resize(cropped_face, (160, 160))

    embedding = embedder.embeddings([resized_face])[0]

    return embedding.tolist()

def check_face_duplicate(new_embedding, face_registry, similarity_threshold=0.3):
    """Check if a face embedding matches any existing registered face"""
    new_emb = np.array(new_embedding)
    for name, registered_embedding in face_registry.items():
        registered_emb = np.array(registered_embedding)
        similarity = np.dot(new_emb, registered_emb) / (
            np.linalg.norm(new_emb) * np.linalg.norm(registered_emb)
        )
        if similarity > (1 - similarity_threshold):
            return name
    return None

def authenticate_face(test_embedding, user_embeddings, similarity_threshold=0.4):
    """Authenticate a face against stored embeddings using vectorized cosine similarity"""
    names = []
    embeddings = []

    for name, embs in user_embeddings.items():
        for emb in embs:
            names.append(name)
            embeddings.append(emb)

    if not embeddings:
        return {
            "match": None,
            "score": 0,
            "authenticated": False
        }

    embeddings = np.array(embeddings)
    test = np.array(test_embedding)

    dot_product = np.dot(embeddings, test)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(test)
    similarities = dot_product / norms

    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > (1 - similarity_threshold):
        return {
            "match": names[best_idx],
            "score": best_score,
            "authenticated": True
        }
    else:
        return {
            "match": None,
            "score": best_score,
            "authenticated": False
        }
