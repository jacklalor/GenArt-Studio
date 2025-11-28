import streamlit as st
import requests
from typing import List

# Backend URL (adjust if needed)
BACKEND_URL = "http://127.0.0.1:8000"


@st.cache_data
def fetch_styles() -> List[str]:
	try:
		res = requests.get(f"{BACKEND_URL}/styles", timeout=5)
		res.raise_for_status()
		data = res.json()
		# Expect list of style objects with 'name' field
		names = [s.get("name", "Unnamed") for s in data]
		return ["None"] + names
	except Exception:
		return ["None"]


def enhance_prompt(prompt: str, style: str) -> dict:
	payload = {"prompt": prompt, "style": style}
	res = requests.post(f"{BACKEND_URL}/enhance_prompt", json=payload, timeout=30)
	res.raise_for_status()
	return res.json()


def main():
	st.set_page_config(page_title="GenArt Studio — Frontend", layout="centered")
	st.title("GenArt Studio")

	st.write("Simple prototype: enter a short description and hit Enhance.")

	styles = fetch_styles()
	col1, col2 = st.columns([3, 1])
	with col1:
		prompt = st.text_area("Prompt", value="A serene landscape with mountains and a river", height=140)
	with col2:
		style = st.selectbox("Style", styles)

	if st.button("Enhance Prompt"):
		if not prompt.strip():
			st.error("Please enter a prompt first.")
		else:
			try:
				with st.spinner("Enhancing prompt with the backend..."):
					result = enhance_prompt(prompt, style)
				enhanced = result.get("enhanced_prompt") or ""
				st.success("Prompt enhanced")
				st.subheader("Enhanced Prompt")
				st.code(enhanced)
				# also show original and style
				st.write("**Original:**", prompt)
				st.write("**Style:**", style)
			except requests.exceptions.RequestException as e:
				st.error(f"API request failed: {e}")


if __name__ == "__main__":
	main()
