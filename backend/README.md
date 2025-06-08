
# SecureVote Face Recognition Backend

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
python main.py
```

The API will be available at http://localhost:8000

## API Endpoints

- POST /register-voter - Register a new voter with face embedding
- POST /authenticate-voter - Authenticate voter using face recognition  
- POST /cast-vote - Record a vote for authenticated voter
- GET /health - Health check

## Usage

The backend replicates the functionality of your Python scripts:
- Uses MTCNN for face detection
- Uses FaceNet for face embeddings
- Stores embeddings as pickle files
- Maintains voted users list
- Cosine similarity for face matching
