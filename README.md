SmartDentalScan
SmartDentalScan is a Flask + PyTorch web app for AI-based dental condition prediction from an intraoral image.

Run (local)
Create venv and install:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Start:
python app.py
Open:
http://127.0.0.1:5000/
Note
Model weights are not committed to Git (e.g., *.pth). Place model.pth locally to run inference.
