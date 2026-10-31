import streamlit as st

from src.explainability.gradcam import gradcam

st.title("Explainability")
st.write("This page uses Grad-CAM only when gradients and patch tokens can be obtained from the ViT encoder. It does not relabel attention maps as Grad-CAM.")
image = st.session_state.get("analysis_image")
bundle = st.session_state.get("analysis_bundle")
result = st.session_state.get("analysis_result")
if image is None or bundle is None or result is None:
    st.info("Run an analysis first on the Skin Analysis page.")
else:
    if st.button("Generate Grad-CAM", type="primary"):
        try:
            with st.spinner("Computing attribution…"):
                original, heatmap, overlay, note = gradcam(bundle, image, result["predicted_index"])
            c1, c2, c3 = st.columns(3)
            c1.image(original, caption="Original", use_container_width=True)
            c2.image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
            c3.image(overlay, caption="Overlay", use_container_width=True)
            st.caption(note)
            st.warning("Attribution maps can be unstable and should not be treated as causal or clinical explanations.")
        except Exception as exc:
            st.error(f"Grad-CAM is unavailable for this loaded model: {exc}")
