import streamlit as st
from PIL import Image

from src.model_loader import load_model
from src.predict import validate_image_bytes, predict_image

st.title("Skin analysis")
st.caption("Inference runs locally using the pretrained model; no uploaded image is intentionally stored.")

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"], help="Maximum size: 10 MB")
if uploaded:
    try:
        image = validate_image_bytes(uploaded.getvalue(), uploaded.type)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.image(image, caption="Input image", use_container_width=True)
        if st.button("Run pretrained model", type="primary"):
            with st.spinner("Loading model and running inference…"):
                try:
                    bundle = load_model()
                    result = predict_image(bundle, image, top_k=3)
                    st.session_state["analysis_image"] = image
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_bundle"] = bundle
                except Exception as exc:
                    st.error(f"Model loading or inference failed: {exc}")

result = st.session_state.get("analysis_result")
if result:
    st.success(f"Top prediction: {result['predicted_label']}")
    st.metric("Predicted class", result["predicted_label"])
    st.subheader("Top three classes")
    for item in result["top"]:
        st.write(f"**{item['label']}** — model score: {item['score']:.2%}")
        st.progress(item["score"])
    st.info("These softmax scores are model scores, not necessarily calibrated diagnostic probabilities. Review the limitations before interpreting them.")
