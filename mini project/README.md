# Patient Readmission & ICU Emergency Risk Prediction

This project is a lightweight web application that demonstrates a medical-scan risk prediction workflow using a Multi-Layer Perceptron (MLP) neural network.

## Features
- Upload a medical scan image
- Preprocess the scan automatically
- Extract texture and intensity features
- Train an MLP model for risk classification
- Evaluate classification performance with standard metrics
- Display prediction outcome in a single-page dashboard

## Tech stack
- Python
- Flask
- NumPy
- Pillow
- scikit-learn

## Run locally
1. Open a terminal in the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open the browser at:
   ```text
   http://localhost:5000
   ```

## Project structure
- `app.py` — Flask app and MLP prediction pipeline
- `templates/index.html` — single-page dashboard interface
- `static/styles.css` — layout and styling
- `requirements.txt` — project dependencies

## Notes
This project uses a synthetic demo dataset to illustrate the full workflow in a self-contained way. It is suitable as a working prototype for academic or demonstration purposes.
