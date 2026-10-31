import streamlit as st

from src.model_loader import load_model, model_metadata

st.set_page_config(page_title="DermaAI | Model information", layout="wide")
st.title("Model and dataset information")
st.caption("Verified from the published Hugging Face model card/configuration and the linked Kaggle dataset page. Publisher claims are clearly separated from independent DermaAI results.")

st.subheader("1. Pretrained model")
st.markdown("""
**Model ID:** [`LaurianeMD/vit-skin-disease`](https://huggingface.co/LaurianeMD/vit-skin-disease)  
**Task:** Multi-class image classification  
**Framework:** PyTorch + Hugging Face Transformers  
**Architecture:** `ViTForImageClassification`  
**Base checkpoint:** `google/vit-base-patch16-224`  
**Model license shown by Hugging Face:** Apache-2.0  
**Model author listed on the card:** Lauriane Mbagdjé Dorenan  
**Published artifact:** `model.safetensors`, float32; the Hugging Face API reports 85,815,574 parameters and approximately 343 MB of model storage.

The model is a Vision Transformer. It divides a 224×224 RGB image into 16×16 patches, processes the patch sequence through transformer self-attention, and sends the class token to a 22-way classification head. The configuration reports 12 encoder layers, 12 attention heads, hidden size 768, intermediate size 3072, GELU activations, and 3 input channels.
""")

st.subheader("2. Preprocessing and inference")
st.markdown("""
The published model card says that images were resized to 224×224, center-cropped, and normalized with ViT base mean/std values. It also mentions optional training-time random flips and rotations. DermaAI uses the model's Hugging Face `AutoImageProcessor`, so the processor configuration bundled with the checkpoint controls runtime preprocessing.

DermaAI selects CUDA when it is available and otherwise falls back to CPU. The loaded model is cached for reuse. The app reports softmax scores as model scores; these are **not necessarily calibrated diagnostic probabilities**.
""")

with st.expander("Publisher-reported training details"):
    st.markdown("""
    The model card reports five epochs, AdamW, cross-entropy loss, GELU activation, float32 precision, and training on a single Google Colab T4 GPU. It does not provide a complete reproducible training configuration such as learning rate, batch size, random seed, augmentation recipe, optimizer betas, weight decay, or exact split-generation code.
    """)

with st.expander("Publisher-reported evaluation — not an independent DermaAI result"):
    st.markdown("""
    The card lists the following epoch history:

    | Epoch | Loss | Accuracy |
    |---:|---:|---:|
    | 1 | 1.4618 | 56.25% |
    | 2 | 0.5954 | 82.90% |
    | 3 | 0.1790 | 95.72% |
    | 4 | 0.0777 | 98.25% |
    | 5 | 0.0613 | 98.49% |

    The card calls 98.49% the final validation accuracy, but does not describe the validation split size, patient-level grouping, class balance, confidence intervals, or per-class results. DermaAI does not reproduce or endorse that number as clinical performance.
    """)

st.subheader("3. Exact configured label mapping")
try:
    metadata = model_metadata(load_model())
    st.write(f"Runtime device: `{metadata['device']}` · CUDA available: `{metadata['cuda_available']}` · Input size: `{metadata['image_size']}`")
    label_rows = [{"Index": idx, "Configured model label": label} for idx, label in metadata["id2label"].items()]
    st.dataframe(label_rows, use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"Runtime model metadata is unavailable until the model can be loaded: {exc}")
    st.write("The published configuration contains 22 labels: Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis, Drug Eruption, Eczema, Infestations Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrh Keratoses, Skin Cancer, Sun Sunlight Damage, Tinea, Unknown Normal, Vascular Tumors, Vasculitis, Vitiligo, and Warts.")

st.subheader("4. Linked dataset")
st.markdown("""
**Dataset:** [`pacificrm/skindiseasedataset`](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)  
**Title:** Skin Disease Dataset  
**Owner shown by Kaggle:** Prashant Kumar Mishra (`pacificrm`)  
**Current Kaggle version in the API metadata:** 6, updated 23 November 2024  
**License shown by Kaggle:** CC0: Public Domain  
**Reported size:** 1,469,587,221 bytes, approximately 1.47 GB  
**Kaggle page summary:** 15.4k files and 22 image classes  
**Stated directory structure:** `SkinDisease/Train/<class>/...` and `SkinDisease/Test/<class>/...`

The dataset page describes a collection of images categorized into 22 classes and lists these names: Acne; Actinic Keratosis; Benign Tumors; Bullous; Candidiasis; Drug Eruption; Eczema; Infestations/Bites; Lichen; Lupus; Moles; Psoriasis; Rosacea; Seborrheic Keratoses; Skin Cancer; Sun/Sunlight Damage; Tinea; Unknown/Normal; Vascular Tumors; Vasculitis; Vitiligo; and Warts.
""")

with st.expander("Dataset-to-model label differences"):
    st.markdown("""
    The dataset page and model configuration use slightly different spellings or separators. Examples include `Infestations/Bites` versus `Infestations Bites`, `Seborrheic Keratoses` versus `Seborrh Keratoses`, `Sun/Sunlight Damage` versus `Sun Sunlight Damage`, and `Unknown/Normal` versus `Unknown Normal`. Evaluation code should map labels deliberately rather than compare raw folder names blindly.
    """)

with st.expander("What the public dataset documentation does not establish"):
    st.markdown("""
    The public dataset page does not provide a complete class-by-class count table, source-image provenance for each class, patient-level identifiers, demographic or skin-tone annotations, dermatologist adjudication details, duplicate-image analysis, or a documented patient-level train/validation/test split procedure. Therefore, dataset size and class names are documented here, but clinical representativeness and leakage risk remain unresolved.
    """)

st.subheader("5. Appropriate use and limitations")
st.markdown("""
- This is a research and educational classifier, not a medical device and not a diagnostic service.
- The model card explicitly says not to use it for diagnosis, treatment, clinical decision-making, or high-stakes applications without domain-expert validation.
- The card notes variable image sources and quality, possible bias across skin tones, lighting, and acquisition devices, limited examples in some classes, and lack of clinician validation.
- Image-only predictions do not use patient history, laboratory values, dermoscopy findings, physical examination, or clinician reasoning.
- A dermatologist or doctor must review and approve any AI classification, laboratory summary, and follow-up suggestion in this prototype.
- Any independent performance claim requires a separately labeled test set, patient-level leakage controls, per-class metrics, calibration analysis, and appropriate clinical validation.
""")

st.info("References: [Hugging Face model card](https://huggingface.co/LaurianeMD/vit-skin-disease) · [Hugging Face configuration](https://huggingface.co/LaurianeMD/vit-skin-disease/raw/main/config.json) · [Hugging Face API metadata](https://huggingface.co/api/models/LaurianeMD/vit-skin-disease) · [Kaggle dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) · [Kaggle dataset API metadata](https://www.kaggle.com/api/v1/datasets/view/pacificrm/skindiseasedataset)")
