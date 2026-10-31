# DermaAI — General Skin Disease Classification MVP

A Streamlit research prototype around the DermaAwaqi-trained recreation of the ViT-Base patch16-224 architecture. The model is trained using the [Kaggle Skin Disease Dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset), reports 97.88% validation accuracy as supplied by the applicant, and uses the [LaurianeMD/vit-skin-disease](https://huggingface.co/LaurianeMD/vit-skin-disease) checkpoint as an architectural/reference source rather than representing the original publisher weights.

The home workspace also includes optional non-identifying history capture, pre-inference image-quality feedback, a laboratory request/result workspace, an opt-in de-identified laboratory-text summarizer, and previous/current image comparison for follow-up documentation.

## Safety and privacy

This is **not a medical device** and must not be used for diagnosis, triage, treatment, or other clinical decisions. Model scores are not necessarily calibrated diagnostic probabilities. Do not upload images without appropriate consent. Uploaded images are processed in memory by this app and are not intentionally stored or sent to an external inference service.

The application’s 97.88% figure is an internal validation result for the applicant-trained recreation, not a clinical-performance claim. It should be accompanied by an independent clinician-labeled test set, leakage audit, per-class metrics, calibration, and subgroup analysis before any clinical use is considered.

## Setup

```bash
cd /home/ubuntu/DermaAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first model load downloads model files from Hugging Face and may require network access. Set `DERMAAI_MODEL_ID` to use a compatible local/cache model for offline operation after its files are available.

## Streamlit Cloud deployment

Deploy from the repository root, with `app.py`, `requirements.txt`, the complete `src/` directory, and the `pages/` directory committed in the same branch. The repository must contain at least:

```text
app.py
requirements.txt
src/__init__.py
src/model_loader.py
src/predict.py
src/explainability/gradcam.py
pages/
```

The app explicitly bootstraps the repository root for imports, but it cannot recover if `src/` was omitted from the deployed repository. After changing `requirements.txt`, use Streamlit Cloud’s **Reboot app** or **Clear cache and restart** so `torch`, `transformers`, and the other dependencies are installed again.

The laboratory analyzer works locally with a conservative rule-based fallback. To enable the optional external OpenAI-compatible summarizer, configure `OPENAI_API_KEY` and optionally `OPENAI_API_BASE` / `OPENAI_MODEL`, then obtain explicit consent in the UI before sending only de-identified laboratory text. No image is sent to that service.

## Evaluation ZIP layout

The Evaluation page accepts a ZIP with one parent folder per ground-truth label:

```text
skin-test.zip
├── Acne/image_001.jpg
├── Acne/image_002.png
└── Eczema/image_003.jpg
```

The app reports accuracy, macro precision, macro recall, macro F1, per-class metrics, and a confusion matrix for that uploaded set only. It does not fabricate a test result.

## Tests

```bash
pytest -q
```

Tests cover file validation and corrupted/unsupported uploads. Model integration depends on downloading the model and is therefore kept separate from the fast unit suite.

## Project layout

- `app.py` — Home page and disclaimer
- `pages/1_Skin_Analysis.py` — upload and top-three inference
- `pages/2_Explainability.py` — Grad-CAM where the ViT hooks are supported
- `pages/3_Model_Information.py` — actual runtime labels and model-card facts
- `pages/4_Evaluation.py` — independent labeled ZIP evaluation
- `src/model_loader.py` — cached-load-ready model and device utilities
- `src/predict.py` — secure in-memory image handling and prediction formatting
- `src/explainability/gradcam.py` — ViT patch-token Grad-CAM
- `tests/` — automated tests

The app intentionally does not issue prescriptions, place laboratory orders, determine that a test is medically necessary, diagnose healing, or provide autonomous treatment recommendations. Image comparison is a review signal sensitive to lighting, distance, focus, and angle; a qualified clinician must assess the patient and original records.

## Known limitations

The model card notes varying image sources/quality, possible bias across skin tones, lighting, and acquisition devices, limited examples in some classes, lack of clinical validation, and uncertain real-world generalization. Grad-CAM is an attribution aid, not a causal or clinical explanation.
