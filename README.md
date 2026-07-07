# Pep2Taste Streamlit Rebuild

This is a Streamlit frontend rebuild of the original Pep2Taste HTML/CSS/JavaScript web application.

## Current status

- Frontend only
- No final model is loaded
- Bitter prediction and umami prediction are separated into two pages
- Mock predictions are used when no backend API URL is configured

## Pages

- Home
- Bitter Prediction
- Umami Prediction
- Database
- Download
- Tools
- Help
- Contact

## Local run

```bash
conda activate BitterUmami
cd /d path\to\pep2taste_streamlit_rebuild
python -m streamlit run app.py
```

## Backend API connection

Create `.streamlit/secrets.toml`:

```toml
BITTER_API_URL = "https://your-bitter-api/predict"
UMAMI_API_URL = "https://your-umami-api/predict"
```

Expected request:

```json
{
  "sequences": ["GLLGFLG"],
  "task": "bitter",
  "threshold": 0.5
}
```

Expected response:

```json
{
  "results": [
    {
      "sequence": "GLLGFLG",
      "length": 7,
      "probability": 0.8123,
      "threshold": 0.5,
      "label": "Bitter",
      "confidence": 0.8123
    }
  ]
}
```

## How to reuse old visualizations

If you want the Database page to show the old interactive HTML plots, copy these folders from the original project into this Streamlit project:

```text
analysis_results/
Database_plot/
```

The app will automatically embed matching HTML files if found.
