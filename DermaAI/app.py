"""DermaAI home page: an original, DermNet-inspired dermatology research workspace."""
from __future__ import annotations

from datetime import date
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Streamlit Cloud normally starts with the repository root on sys.path, but an
# explicit bootstrap makes deployments robust when the app is launched from a
# different working directory. The src/ directory and its __init__.py must be
# committed to the same repository as this file.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.model_loader import load_model
    from src.predict import assess_image_quality, compare_followup_images, predict_image, validate_image_bytes
except ModuleNotFoundError as exc:
    # Dependency errors are re-raised unchanged so Streamlit logs identify the
    # missing package (for example torch or transformers).
    if exc.name != "src":
        raise
    SRC_ROOT = PROJECT_ROOT / "src"
    if SRC_ROOT.is_dir():
        sys.path.insert(0, str(SRC_ROOT))
        from model_loader import load_model
        from predict import assess_image_quality, compare_followup_images, predict_image, validate_image_bytes
    else:
        raise

st.set_page_config(page_title="DermaAI | Home", page_icon=None, layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Original visual system: calm clinical colors, generous whitespace, card layout.
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root { --ink:#132b43; --muted:#536779; --teal:#087f8c; --teal-dark:#075b66; --mint:#e7f6f4; --line:#d5e1e8; --warm:#fff8ee; }
[data-testid="stAppViewContainer"] { background: #fbfdfe; }
[data-testid="stHeader"] { background: rgba(251,253,254,.9); }
.block-container { max-width: 1280px; padding-top: 1.3rem; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stMarkdown, .stText, label, [data-testid="stWidgetLabel"] { color:var(--ink); }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--muted); }
.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.15rem; }
.brand-mark { width:42px; height:42px; border-radius:13px; background:var(--teal); color:white; display:flex; align-items:center; justify-content:center; font-weight:800; letter-spacing:-1px; }
.brand-name { color:var(--ink); font-size:1.85rem; font-weight:850; letter-spacing:-.05em; }
.brand-sub { color:var(--muted); font-size:.82rem; margin-left:3.2rem; margin-top:-.3rem; }
.hero { background:linear-gradient(115deg,#eaf7f5 0%,#f4fbfa 56%,#fff 100%); border:1px solid #d6ece9; border-radius:24px; padding:2.1rem 2.2rem; margin:1.25rem 0 1.2rem; }
.hero h1 { color:var(--ink); font-size:2.8rem; line-height:1.06; margin:.25rem 0 .65rem; letter-spacing:-.055em; }
.hero p { color:#456477; max-width:770px; font-size:1.04rem; line-height:1.6; }
.eyebrow { color:var(--teal-dark); font-size:.74rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
.card { background:#fff; border:1px solid var(--line); border-radius:17px; padding:1.1rem 1.15rem; min-height:150px; box-shadow:0 5px 18px rgba(28,58,77,.035); }
.card h3 { color:var(--ink); font-size:1.08rem; margin:.25rem 0 .42rem; }
.card p { color:var(--muted); font-size:.9rem; line-height:1.48; }
.pill { display:inline-block; background:var(--mint); color:var(--teal-dark); border-radius:99px; padding:.22rem .6rem; font-size:.73rem; font-weight:750; }
.section-title { color:var(--ink); margin-top:1.6rem; margin-bottom:.35rem; }
.small-muted { color:var(--muted); font-size:.88rem; }
.notice { background:var(--warm); border:1px solid #f1dfbf; border-radius:14px; padding:.9rem 1rem; color:#795b28; font-size:.9rem; }
.quality-good { background:#eaf7ef; border:1px solid #bde1c9; border-radius:12px; padding:.75rem .9rem; color:#21683b; }
.quality-warn { background:#fff7e9; border:1px solid #efd49b; border-radius:12px; padding:.75rem .9rem; color:#805c18; }
.nav-label { color:var(--muted); font-size:.78rem; font-weight:700; text-align:center; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:.7rem .85rem; border-radius:14px; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state: intentionally minimal, in-memory, and not patient-identifying.
# -----------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "case_id": "",
    "history_saved": False,
    "lab_requests": [],
    "lab_results": [],
    "followups": [],
    "analysis_result": None,
    "analysis_image": None,
    "analysis_bundle": None,
    "previous_image": None,
    "current_followup_image": None,
    "followup_comparison": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def local_lab_review(text: str) -> dict[str, str]:
    """Conservative local parser used when no external AI service is enabled."""
    lower = text.lower()
    flags = []
    for token in ("critical", "urgent", "positive", "abnormal", "elevated", "high", "low", "decreased", "indeterminate"):
        if token in lower:
            flags.append(token)
    return {
        "summary": "The report was parsed for explicit words and should be reviewed against its original reference intervals.",
        "signals": ", ".join(sorted(set(flags))) if flags else "No explicit flag words detected",
        "next_step": "A qualified clinician should interpret the original report, units, reference range, specimen quality, and clinical context.",
    }


def external_lab_review(text: str) -> dict[str, str] | None:
    """Use an explicitly configured OpenAI-compatible endpoint only after consent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    endpoint = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a safety-constrained laboratory report summarizer. Do not diagnose, prescribe, or recommend treatment. Return JSON with keys summary, signals, missing_context, urgency_for_clinician_review, and next_step. Mention that a clinician must interpret the original report."},
            {"role": "user", "content": "Summarize this de-identified laboratory text conservatively:\n" + text[:12000]},
        ],
    }
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode())
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return None

# The Streamlit multipage navigation is intentionally left as the only navigation.
# This avoids duplicating the framework's own page list in the sidebar.
st.markdown('<div class="brand"><div class="brand-mark">D</div><div class="brand-name">DermaAI</div></div><div class="brand-sub">Home · Dermatology research workspace</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="eyebrow">Evidence-aware skin image research</div><h1>Understand skin findings.<br>Document the clinical context.</h1><p>DermaAI combines local image classification with a structured, clinician-reviewable case workspace. Add symptom history, exposures, medicines, laboratory notes, and follow-up plans when they are relevant to the research case.</p></div>',
    unsafe_allow_html=True,
)
hero_image = Path(__file__).parent / "assets" / "dermaai-hero.png"
if hero_image.exists():
    st.image(str(hero_image), caption="A visual research aid — not a clinical finding", use_container_width=True)

st.warning("Research prototype only. The model does not diagnose, triage, prescribe, or replace a dermatologist. Seek urgent clinical care for rapidly worsening symptoms, breathing difficulty, facial swelling, extensive blistering, severe pain, fever, or mucosal involvement.")

# -----------------------------------------------------------------------------
# Public-resource style discovery area.
# -----------------------------------------------------------------------------
st.markdown('<h2 class="section-title">Explore the dermatology workspace</h2>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">An original interface inspired by the way public dermatology resources organize topics, images, cases, education, and professional review.</div>', unsafe_allow_html=True)

cards = [
    ("Analyze an image", "Upload a JPG, JPEG, or PNG and review the pretrained model’s top classes with an explicit uncertainty boundary.", "Start analysis"),
    ("Skin topics A–Z", "Browse common condition categories, diagnosis-and-testing concepts, infections, rashes, pigmentary disorders, hair, and nails.", "Open topics"),
    ("Clinical case notes", "Capture symptom timeline, distribution, exposures, previous treatments, relevant history, and patient-reported concerns.", "Add history"),
    ("Laboratory workspace", "Record clinician-requested tests and enter results for structured review. The app does not issue prescriptions or orders.", "Review labs"),
    ("Follow-up planning", "Document a review date, monitoring questions, response, adverse effects, and escalation instructions for clinician sign-off.", "Plan follow-up"),
    ("Learning and evidence", "Open the model information, Grad-CAM, independent evaluation, and links to reputable dermatology resources.", "View resources"),
]
cols = st.columns(3)
for i, (title, text, action) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f'<div class="card"><span class="pill">{action}</span><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Single-page case workspace: optional context before image analysis.
# -----------------------------------------------------------------------------
st.markdown('<h2 class="section-title">Case workspace</h2>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Use a non-identifying case label. Do not enter names, addresses, phone numbers, or other unnecessary personal information.</div>', unsafe_allow_html=True)

case_tab, image_tab, labs_tab, follow_tab = st.tabs(["1. History & context", "2. Image analysis", "3. Labs", "4. Follow-up"])

with case_tab:
    with st.form("history_form"):
        top = st.columns(3)
        with top[0]:
            st.text_input("Non-identifying case label", key="case_id", placeholder="e.g., study-014")
        with top[1]:
            age_group = st.selectbox("Age group", ["Not recorded", "Child", "Adolescent", "Adult", "Older adult"])
        with top[2]:
            skin_site = st.multiselect("Anatomical site", ["Face", "Scalp", "Trunk", "Arms", "Hands", "Legs", "Feet", "Genital", "Mucosal", "Generalised", "Other"])

        c1, c2 = st.columns(2)
        with c1:
            onset = st.selectbox("Onset", ["Not recorded", "Sudden", "Gradual", "Recurrent", "Present since birth"])
            duration = st.text_input("Duration", placeholder="e.g., 3 weeks")
            symptoms = st.multiselect("Symptoms", ["Itch", "Pain", "Burning", "Bleeding", "Scaling", "Blistering", "Ulceration", "Hair loss", "Nail change", "No symptoms", "Other"])
            morphology = st.text_area("Observed features", placeholder="Colour, border, scale, crust, vesicles, nodules, symmetry, change over time…")
        with c2:
            exposures = st.text_area("Relevant exposures", placeholder="New cosmetics, workplace chemicals, plants, travel, animals, contacts, sun, bites…")
            medicines = st.text_area("Medicines and allergies", placeholder="Current/recent medicines, topical products, known allergies…")
            medical_history = st.text_area("Relevant medical history", placeholder="Atopy, autoimmune disease, immunosuppression, cancer history, pregnancy, family history…")
            patient_concern = st.text_area("Patient-reported concern or goal", placeholder="What changed? What would the patient like clarified?")

        red_flags = st.multiselect("Safety screen — clinician review", ["Rapid progression", "Fever or systemic symptoms", "Mucosal involvement", "Severe pain", "Facial swelling", "Breathing difficulty", "Extensive blistering", "None reported", "Not assessed"])
        submitted = st.form_submit_button("Save context for this session", type="primary")
        if submitted:
            st.session_state["history_saved"] = True
            st.session_state["history"] = {
                "age_group": age_group, "site": skin_site, "onset": onset, "duration": duration,
                "symptoms": symptoms, "morphology": morphology, "exposures": exposures,
                "medicines": medicines, "medical_history": medical_history,
                "patient_concern": patient_concern, "red_flags": red_flags,
            }
            st.success("Context saved in this session only. Review and correct it before any clinician-facing use.")
    if st.session_state.get("history_saved"):
        history = st.session_state.get("history", {})
        st.caption(f"Saved fields: {sum(bool(v) for v in history.values())}/{len(history)} non-identifying sections.")

with image_tab:
    st.subheader("Image analysis")
    st.write("The classifier uses the image only. History fields are stored as structured context for review and are not passed into this image-only model.")
    lab_status = st.radio("Laboratory context", ["No laboratory investigation planned", "Laboratory requested — result pending", "Laboratory result available for clinician review"], horizontal=True)
    lab_pending = lab_status == "Laboratory requested — result pending"
    if lab_pending:
        st.warning("A laboratory investigation is marked as pending. Bring or enter the laboratory result in the Labs tab before relying on the image review. The image model cannot substitute for a required laboratory test.")
    uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"], key="home_image")
    if uploaded:
        try:
            image = validate_image_bytes(uploaded.getvalue(), uploaded.type)
            st.image(image, caption="Input image", width=420)
            quality = assess_image_quality(image)
            q1, q2, q3 = st.columns(3)
            q1.metric("Resolution", f"{quality['width']} × {quality['height']} px")
            q2.metric("Image size", f"{quality['megapixels']} MP")
            q3.metric("Focus signal", f"{quality['focus_score']}")
            if quality["issues"]:
                st.markdown('<div class="quality-warn"><strong>Image quality warning</strong><br>' + "<br>".join(quality["issues"]) + '<br><br>For a better prediction, upload the highest-resolution, sharply focused image available. Keep the lesion centered, well lit, and captured at a consistent distance.</div>', unsafe_allow_html=True)
                run_anyway = st.checkbox("Run the model anyway (quality limitations understood)", key="home_quality_override")
            else:
                st.markdown('<div class="quality-good"><strong>Image quality looks suitable for a research preview.</strong> This is not a guarantee of diagnostic reliability.</div>', unsafe_allow_html=True)
                run_anyway = True
            if st.button("Run local model inference", type="primary", key="home_run", disabled=lab_pending or (bool(quality["issues"]) and not run_anyway)):
                with st.spinner("Loading model and running inference…"):
                    bundle = load_model()
                    result = predict_image(bundle, image, top_k=3)
                    st.session_state["analysis_image"] = image
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_bundle"] = bundle
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Model loading or inference failed: {exc}")

    result = st.session_state.get("analysis_result")
    if result:
        st.success(f"Model top class: {result['predicted_label']}")
        metric_cols = st.columns(3)
        for col, item in zip(metric_cols, result["top"]):
            with col:
                st.metric(item["label"], f"{item['score']:.1%}")
        st.info("Scores are not calibrated diagnostic probabilities. Treat the output as a research signal requiring clinical examination and appropriate testing.")
        st.warning("Dermatologist or doctor approval is required before this AI classification is used in a clinical or research decision.")
        image_approval = st.selectbox("Clinician review status", ["Pending dermatologist/doctor review", "Reviewed — not approved for use", "Approved by dermatologist/doctor"], key="image_approval")
        if image_approval == "Approved by dermatologist/doctor":
            st.success("Approval recorded for this session. Keep the original image, history, and clinical reasoning with the case record.")
        st.page_link("pages/2_Explainability.py", label="Open Grad-CAM explainability", icon=None)

with labs_tab:
    st.subheader("Laboratory request and result workspace")
    st.markdown('<div class="notice">This section documents clinician-selected investigations. It does not generate a prescription, place an order, or determine whether a test is medically necessary.</div>', unsafe_allow_html=True)
    st.write("")
    lab_form = st.form("lab_request_form")
    with lab_form:
        c1, c2, c3 = st.columns(3)
        with c1:
            test_name = st.text_input("Investigation", placeholder="e.g., fungal microscopy / culture")
        with c2:
            reason = st.text_input("Clinical question", placeholder="What question should this test address?")
        with c3:
            urgency = st.selectbox("Priority", ["Routine", "Time-sensitive — clinician review", "Urgent — follow local protocol"])
        specimen = st.text_input("Specimen / collection note", placeholder="e.g., skin scraping from active border")
        if lab_form.form_submit_button("Add clinician-requested investigation"):
            if test_name.strip():
                st.session_state["lab_requests"].append({"Investigation": test_name.strip(), "Clinical question": reason, "Priority": urgency, "Specimen": specimen})
                st.success("Investigation note added to this session.")
            else:
                st.error("Enter an investigation name or leave this section blank.")

    if st.session_state["lab_requests"]:
        st.dataframe(pd.DataFrame(st.session_state["lab_requests"]), use_container_width=True, hide_index=True)

    result_form = st.form("lab_result_form")
    with result_form:
        c1, c2, c3 = st.columns(3)
        with c1:
            result_name = st.text_input("Result / test name", placeholder="e.g., KOH microscopy")
        with c2:
            result_value = st.text_input("Reported result", placeholder="Paste the laboratory report wording")
        with c3:
            result_flag = st.selectbox("Reported flag", ["Not assessed", "Within stated range", "Above range", "Below range", "Positive", "Negative", "Indeterminate"])
        lab_comment = st.text_area("Clinician interpretation note", placeholder="Record the laboratory's reference range, specimen quality, and clinical correlation needed. Do not rely on this app for interpretation.")
        if result_form.form_submit_button("Add result for review"):
            if result_name.strip() and result_value.strip():
                st.session_state["lab_results"].append({"Test": result_name.strip(), "Reported result": result_value, "Flag": result_flag, "Review note": lab_comment})
                st.success("Laboratory result added to this session.")
            else:
                st.error("Enter both a result/test name and the reported result.")
    if st.session_state["lab_results"]:
        st.dataframe(pd.DataFrame(st.session_state["lab_results"]), use_container_width=True, hide_index=True)
        st.caption("Laboratory reports can be context-dependent. A qualified clinician should interpret the original report, reference intervals, specimen, and patient context.")

    st.subheader("AI-assisted lab report review")
    st.write("Paste a de-identified report or result block for a conservative summary of explicit flags, missing context, and questions for a clinician. This tool does not diagnose, prescribe, or decide whether a laboratory test is needed.")
    lab_text = st.text_area("De-identified laboratory text", height=160, placeholder="Include the test name, result, units, reference interval, specimen, and the laboratory's own flags where available.")
    allow_external = st.checkbox("I consent to send this de-identified text to the configured AI service for summarization", help="If no AI service is configured, the app uses a local rule-based review and sends nothing externally.")
    if st.button("Analyze laboratory text", key="analyze_lab"):
        if not lab_text.strip():
            st.error("Paste a laboratory result first.")
        elif allow_external and os.getenv("OPENAI_API_KEY"):
            with st.spinner("Summarizing the de-identified report…"):
                analysis = external_lab_review(lab_text)
            if analysis is None:
                st.warning("The configured AI service was unavailable. Showing the local conservative review instead.")
                analysis = local_lab_review(lab_text)
            else:
                st.success("AI-assisted summary completed. Verify it against the original laboratory report.")
            st.session_state["last_lab_analysis"] = analysis
        else:
            if allow_external and not os.getenv("OPENAI_API_KEY"):
                st.info("No external AI service is configured, so the report stayed local.")
            st.session_state["last_lab_analysis"] = local_lab_review(lab_text)
    if st.session_state.get("last_lab_analysis"):
        st.json(st.session_state["last_lab_analysis"])
        st.warning("A doctor must approve the AI-assisted laboratory summary against the original report before it informs the case.")
        lab_approval = st.selectbox("Laboratory AI review status", ["Pending doctor review", "Reviewed — not approved", "Approved by doctor"], key="lab_approval")
        if lab_approval == "Approved by doctor":
            st.success("Doctor approval recorded for this session.")

with follow_tab:
    st.subheader("Patient follow-up and monitoring")
    st.markdown('<div class="notice">Follow-up entries are documentation prompts for a qualified clinician or research coordinator. They are not automated treatment instructions.</div>', unsafe_allow_html=True)
    st.write("Upload comparable captures when possible: same body site, distance, lighting, orientation, and a scale reference. The comparison is a visual change signal, not a healing detector.")
    previous_upload = st.file_uploader("Previous lesion image", type=["jpg", "jpeg", "png"], key="previous_lesion")
    current_upload = st.file_uploader("Current lesion image", type=["jpg", "jpeg", "png"], key="current_lesion")
    if previous_upload and current_upload:
        try:
            previous_image = validate_image_bytes(previous_upload.getvalue(), previous_upload.type)
            current_image = validate_image_bytes(current_upload.getvalue(), current_upload.type)
            pcol, ccol = st.columns(2)
            pcol.image(previous_image, caption="Previous capture", use_container_width=True)
            ccol.image(current_image, caption="Current capture", use_container_width=True)
            if st.button("Compare previous and current captures", type="primary", key="compare_lesion"):
                st.session_state["previous_image"] = previous_image
                st.session_state["current_followup_image"] = current_image
                st.session_state["followup_comparison"] = compare_followup_images(previous_image, current_image)
        except ValueError as exc:
            st.error(str(exc))
    comparison = st.session_state.get("followup_comparison")
    if comparison:
        st.subheader("Comparison review")
        f1, f2 = st.columns(2)
        f1.metric("Visual change signal", comparison["signal"])
        f2.metric("Change score", comparison["change_score"])
        st.info("The app cannot determine whether a lesion is healing from pixels alone. A lower change score suggests visual stability; it does not prove healing. A higher score may reflect healing, progression, or different capture conditions.")
        st.markdown(f"**Suggested next action for clinician review:** {comparison['action']}")
        st.caption(comparison["limitation"])
        suggested_days = 30 if comparison["signal"] == "little visible change" else 14 if comparison["signal"] == "some visible change" else 7
        suggested_date = date.today().fromordinal(date.today().toordinal() + suggested_days)
        st.markdown(f"**AI-assisted follow-up suggestion:** consider a clinician review in about {suggested_days} days ({suggested_date.isoformat()}). This is only a scheduling prompt; the doctor decides the actual interval.")
        followup_approval = st.selectbox("Doctor approval of next follow-up suggestion", ["Pending doctor approval", "Approved by doctor", "Doctor selected a different interval"], key="followup_approval")
        if followup_approval == "Approved by doctor":
            st.success("Doctor approval recorded for the suggested follow-up interval.")
        clinician_assessment = st.selectbox("Clinician/reviewer assessment", ["Not assessed", "Appears improved", "Appears unchanged", "Appears worse", "Uncertain — needs examination"], key="clinician_assessment")
        if clinician_assessment == "Appears improved":
            st.success("Document the features supporting improvement and keep the planned review unless symptoms or red flags require earlier care.")
        elif clinician_assessment == "Appears worse":
            st.warning("Prioritise clinician assessment sooner and follow local escalation protocol, especially with pain, bleeding, ulceration, fever, or rapid change.")
        elif clinician_assessment in {"Uncertain — needs examination", "Not assessed"}:
            st.info("Do not label the lesion as healing yet. Arrange appropriate clinical examination and compare the original images and report.")
    with st.form("followup_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            review_date = st.date_input("Planned review date", value=date.today())
        with c2:
            owner = st.text_input("Responsible reviewer", placeholder="Role or study code — no personal contact data")
        with c3:
            status = st.selectbox("Case status", ["Needs clinician review", "Awaiting investigation", "Monitoring", "Closed by reviewer", "Escalate per local protocol"])
        monitoring = st.multiselect("Monitor at follow-up", ["Size / number", "Colour", "Border", "Pain", "Itch", "Bleeding", "Ulceration", "Function", "Treatment adverse effects", "Patient concern"])
        plan = st.text_area("Clinician-entered follow-up plan", placeholder="Document the next review question, records to obtain, and escalation instructions according to local protocol.")
        if st.form_submit_button("Save follow-up entry"):
            st.session_state["followups"].append({"Review date": review_date.isoformat(), "Reviewer role": owner, "Status": status, "Monitor": ", ".join(monitoring), "Plan": plan})
            st.success("Follow-up entry saved in this session.")
    if st.session_state["followups"]:
        st.dataframe(pd.DataFrame(st.session_state["followups"]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Topic/resource directory. These are links to external educational sources,
# not copied medical content.
# -----------------------------------------------------------------------------
st.markdown('<h2 class="section-title">Dermatology learning directory</h2>', unsafe_allow_html=True)
resource_cols = st.columns(4)
resources = [
    ("Topics A–Z", "https://dermnetnz.org/topics", "Conditions, symptoms, treatments, and diagnosis topics."),
    ("Clinical images", "https://dermnetnz.org/images", "Visual learning resource with image licensing caveats."),
    ("Cases", "https://dermnetnz.org/cases", "Case-based learning and differential thinking."),
    ("Editorial process", "https://dermnetnz.org/about/editorial-process", "How an authoritative dermatology resource reviews content."),
]
for col, (title, url, description) in zip(resource_cols, resources):
    with col:
        st.markdown(f"**[{title}]({url})**")
        st.caption(description)

st.divider()
st.caption("DermaAI is an independent research prototype. DermNet is an external educational resource; this interface does not reproduce or imply endorsement of DermNet content. No uploaded image or case data is intentionally persisted by this app.")
