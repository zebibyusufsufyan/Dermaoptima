import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.model_loader import load_model
from src.predict import validate_image_bytes, predict_image

st.title("Independent evaluation")
st.write("Provide a separate labeled test set as a folder archive. Expected layout: `class_name/image.jpg`. Results below are computed only from the uploaded test set; no metrics are fabricated.")
archive = st.file_uploader("Upload a ZIP test set", type=["zip"])
if archive and st.button("Evaluate test set", type="primary"):
    try:
        bundle = load_model()
        y_true, y_pred, rows = [], [], []
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not members:
                raise ValueError("The ZIP contains no JPG, JPEG, or PNG files.")
            for member in members:
                parts = Path(member.filename).parts
                if len(parts) < 2 or parts[-2].startswith("."):
                    continue
                true_label = parts[-2]
                data = zf.read(member)
                image = validate_image_bytes(data)
                pred = predict_image(bundle, image, top_k=1)["predicted_label"]
                y_true.append(true_label); y_pred.append(pred)
                rows.append({"file": member.filename, "true": true_label, "predicted": pred})
        if not y_true:
            raise ValueError("No images with a parent-folder label were found.")
        labels = sorted(set(y_true) | set(y_pred))
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        st.session_state["eval_rows"] = rows
        st.session_state["eval_report"] = report
        st.session_state["eval_labels"] = labels
        st.session_state["eval_cm"] = confusion_matrix(y_true, y_pred, labels=labels)
        st.success(f"Evaluated {len(y_true)} images across {len(set(y_true))} observed true classes.")
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

report = st.session_state.get("eval_report")
if report:
    macro = report["macro avg"]
    st.subheader("Independent metrics")
    st.write({"accuracy": accuracy_score([r["true"] for r in st.session_state["eval_rows"]], [r["predicted"] for r in st.session_state["eval_rows"]]), "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1-score"]})
    st.subheader("Per-class metrics")
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(st.session_state["eval_cm"], cmap="Blues")
    labels = st.session_state["eval_labels"]
    ax.set_xticks(range(len(labels)), labels, rotation=90); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("These are independent results for the uploaded ZIP only. They must not be conflated with the original model-card validation result.")
