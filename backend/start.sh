
#!/bin/bash
echo "Starting SecureVote Face Recognition Backend..."
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting FastAPI server..."
python main.py
