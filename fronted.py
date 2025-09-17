import streamlit as st
import requests

st.set_page_config(page_title="News Bias Detector", layout="centered")

st.title("📰 News Bias Detection ")
st.markdown("Paste a news article and Check Biased and Un-Biased News .")

# Sidebar for API URL
with st.sidebar:
    st.header("Settings")
    api_url = "http://127.0.0.1:8000/predict/"

# Text input
input_text = st.text_area("Paste article text here", height=250)

# Predict button
if st.button("Predict"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    elif not api_url.strip():
        st.error("Please provide FastAPI URL in the sidebar.")
    else:
        try:
            resp = requests.post(api_url, json={"text": input_text})
            st.balloons()
            resp.raise_for_status()
            resp_json = resp.json()
            pred_num = resp_json.get("bias_prediction", "N/A")
            pred_label = resp_json.get("label", "N/A")
            conf = resp_json.get("confidence", None)

            st.success(f"New's Prediction is :- {pred_label} - ({pred_num})")
            if conf is not None:
                st.info(f"Confidence: {conf:.2f}")
        except Exception as e:
            st.error(f"API request failed: {e}")