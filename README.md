# Fake News Detector

Lightweight Streamlit app that analyzes news articles and predicts whether they are Real or Fake using a trained ML model.

**Features**
- Binary Real / Fake prediction with confidence score
- Text statistics (words, characters, sentences, reading time)
- Fake-news indicator flags (ALL CAPS, sensational language, punctuation, links)
- Improved UI with visual confidence bar and recommendations

**Project Files**
- `app.py` — Streamlit application (main)
- `test_app.py` — debug version that shows model-load steps
- `True.csv`, `Fake.csv` — original datasets
- `True_cleaned.csv`, `Fake_cleaned.csv` — cleaned datasets
- `vectorizer.jb`, `lr_model.jb` — model artifacts (required at runtime)

## Requirements
- Python 3.8+
- streamlit
- joblib
- scikit-learn
- pandas

Install dependencies (recommended inside a virtualenv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install streamlit joblib scikit-learn pandas
```

## Run the app

From the project root:

```powershell
streamlit run app.py
```

Open http://localhost:8501 in your browser if it does not open automatically.

## Notes
- Ensure `vectorizer.jb` and `lr_model.jb` are present in the project root before starting the app.
- If the app shows no output, try running `test_app.py` to get verbose model-loading errors:

```powershell
streamlit run test_app.py
```

## Data cleaning
Use `clean_all_csv.py` to regenerate cleaned CSVs from the raw datasets:

```powershell
python clean_all_csv.py
```

## License & Attribution
This repository contains example code and cleaned datasets for educational purposes.