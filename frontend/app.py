# Import required libraries
import streamlit as st  # Web framework for building the UI
import requests  # For making HTTP requests to the backend API
from typing import List  # Type hints for better code clarity
import base64  # For decoding base64-encoded images from API
from io import BytesIO  # For handling image data in memory
from PIL import Image  # Python Imaging Library for image processing
  
# Backend API configuration
# This is where the FastAPI backend server is running
BACKEND_URL = "http://127.0.0.1:8000"


@st.cache_data  # Cache the styles to avoid repeated API calls
def fetch_styles() -> List[str]:
	"""Fetch available art styles from the backend API.
	
	Returns:
		List of style names, with "None" as the first option
	"""
	try:
		# Make GET request to backend /styles endpoint
		res = requests.get(f"{BACKEND_URL}/styles", timeout=5)
		res.raise_for_status()  # Raise error if request failed
		data = res.json()  # Parse JSON response
		
		# Extract style names from response (expects list of objects with 'name' field)
		names = [s.get("name", "Unnamed") for s in data]
		
		# Add "None" as first option for no style selection
		return ["None"] + names
	except Exception:
		# If API call fails, return just "None" as fallback
		return ["None"]


def enhance_prompt(prompt: str, style: str) -> dict:
	"""Send user's prompt to backend for AI enhancement using GPT-4o mini.
	
	Args:
		prompt: User's original text description
		style: Selected art style (e.g., "Photorealistic", "Anime")
		
	Returns:
		Dict containing original_prompt, enhanced_prompt, and style
	"""
	# Prepare request payload with prompt and style
	payload = {"prompt": prompt, "style": style}
	
	# Send POST request to backend enhancement endpoint
	res = requests.post(f"{BACKEND_URL}/enhance_prompt", json=payload, timeout=30)
	res.raise_for_status()  # Raise error if request failed
	
	# Return JSON response containing enhanced prompt
	return res.json()


def generate_image(prompt: str, style: str, model: str) -> dict:
	"""Generate an image using the backend API.
	
	Args:
		prompt: Text description of image to generate
		style: Selected art style
		model: Generation mode ("demo" for instant, "stable-diffusion" for AI)
		
	Returns:
		Dict containing base64-encoded image, prompt, and style
	"""
	# Prepare request payload with all generation parameters
	payload = {"prompt": prompt, "style": style, "model": model}
	
	# Send POST request to backend generation endpoint
	# Timeout set to 10 minutes for Stable Diffusion (can take 60-120 seconds)
	res = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=600)
	res.raise_for_status()  # Raise error if request failed
	
	# Return JSON response containing base64-encoded image
	return res.json()


def main():
	"""Main application function that builds the Streamlit UI."""
	
	# Configure the Streamlit page settings
	st.set_page_config(
		page_title="GenArt Studio | AI-Powered Art Generation",
		layout="centered",  # Center the content on the page
		page_icon="🎨"  # Display art palette emoji in browser tab
	)
	
	# Initialize session state for storing image generation history
	# This persists across reruns within the same browser session
	if 'history' not in st.session_state:
		st.session_state.history = []  # Empty list to store generated images
	
	# Display main title
	st.title("GenArt Studio")

	# Display description text
	st.write("Transform your ideas into stunning AI-generated artwork. Enter a description and select your preferred style.")

	# Fetch available styles from backend API
	styles = fetch_styles()
	
	# Create 3-column layout for input controls
	# Column widths: 3 parts for prompt, 1 part each for style and model
	col1, col2, col3 = st.columns([3, 1, 1])
	
	# Column 1: Text area for user's prompt/description
	with col1:
		prompt = st.text_area(
			"Prompt",
			value="A serene landscape with mountains and a river",  # Default example
			height=140
		)
	
	# Column 2: Dropdown for selecting art style
	with col2:
		style = st.selectbox("Style", styles)
	
	# Column 3: Dropdown for selecting generation model
	with col3:
		model = st.selectbox(
			"Model",
			["demo", "stable-diffusion"],
			help="Demo: Instant generation for testing\nStable Diffusion: High-quality AI generation (60-120s)"
		)

	# Create 2-column layout for action buttons
	col_btn1, col_btn2 = st.columns(2)
	
	# Column 1: Enhance Prompt button (uses GPT-4o mini to improve prompt)
	with col_btn1:
		if st.button("Enhance Prompt"):
			# Validate that user entered a prompt
			if not prompt.strip():
				st.error("Please enter a prompt first.")
			else:
				try:
					# Show loading spinner while API processes request
					with st.spinner("Enhancing prompt with the backend..."):
						result = enhance_prompt(prompt, style)
					
					# Extract enhanced prompt from API response
					enhanced = result.get("enhanced_prompt") or ""
					
					# Display success message and results
					st.success("Prompt enhanced")
					st.subheader("Enhanced Prompt")
					st.code(enhanced)  # Display enhanced prompt in code block
					st.write("**Original:**", prompt)
					st.write("**Style:**", style)
				except requests.exceptions.RequestException as e:
					# Handle API errors (network issues, timeouts, etc.)
					st.error(f"API request failed: {e}")
	
	# Column 2: Generate Image button (primary action)
	with col_btn2:
		if st.button("Generate Image", type="primary"):
			# Validate that user entered a prompt
			if not prompt.strip():
				st.error("Please enter a prompt first.")
			else:
				try:
					# Show different loading messages based on selected model
					if model == "demo":
						# Demo mode: instant procedural generation
						with st.spinner("Generating image..."):
							result = generate_image(prompt, style, model)
					else:
						# Stable Diffusion: takes 60-120 seconds
						with st.spinner("Generating high-quality AI image (60-120 seconds)..."):
							result = generate_image(prompt, style, model)
					
					# Extract base64-encoded image from API response
					image_b64 = result.get("image")
					
					if image_b64:
						# Decode base64 string to image bytes
						image_bytes = base64.b64decode(image_b64)
						
						# Open image from bytes using PIL
						image = Image.open(BytesIO(image_bytes))
						
						# Add generated image to session history (newest first)
						st.session_state.history.insert(0, {
							'image_bytes': image_bytes,  # Store raw bytes for download
							'prompt': result.get("prompt"),  # Original prompt text
							'style': result.get("style"),  # Selected style
							'model': model  # Model used (demo or stable-diffusion)
						})
						
						# Display success message and generated image
						st.success("Image generated!")
						st.image(image, caption=f"Generated: {result.get('prompt')}", use_container_width=True)
						
						# Add download button for saving the image
						st.download_button(
							label="⬇️ Download Image",
							data=image_bytes,
							# Create filename from first 30 chars of prompt
							file_name=f"genart_{prompt[:30].replace(' ', '_')}.png",
							mime="image/png"
						)
						
						# Display metadata about the generation
						st.write("**Prompt used:**", result.get("prompt"))
						st.write("**Style:**", result.get("style"))
					else:
						st.error("No image returned from backend")
						
				except requests.exceptions.RequestException as e:
					# Handle network/API errors
					st.error(f"Image generation failed: {e}")
				except Exception as e:
					# Handle any other errors (image decoding, etc.)
					st.error(f"Error displaying image: {e}")
	
	# Gallery section - displays all previously generated images
	if st.session_state.history:
		st.divider()  # Visual separator from generation section
		st.subheader("Generation History")
		
		# Create layout for gallery header with clear button
		col_left, col_right = st.columns([3, 1])
		with col_right:
			# Button to clear all history
			if st.button("🗑️ Clear History"):
				st.session_state.history = []  # Empty the history list
				st.rerun()  # Reload page to reflect changes
		
		# Display images in a responsive 3-column grid
		cols_per_row = 3
		
		# Iterate through history in chunks of 3 (one row at a time)
		for i in range(0, len(st.session_state.history), cols_per_row):
			cols = st.columns(cols_per_row)  # Create 3 columns for this row
			
			# Fill each column in the row
			for j in range(cols_per_row):
				idx = i + j  # Calculate index in history list
				
				# Only display if we have an item at this index
				if idx < len(st.session_state.history):
					item = st.session_state.history[idx]
					
					with cols[j]:
						# Open image from stored bytes
						img = Image.open(BytesIO(item['image_bytes']))
						
						# Display image scaled to column width
						st.image(img, use_container_width=True)
						
						# Display full prompt text as caption
						st.caption(f"**{item['prompt']}**")
						
						# Display style and model metadata
						st.caption(f"Style: {item['style']} | Model: {item['model']}")
						
						# Individual download button for this image
						st.download_button(
							label="⬇️",
							data=item['image_bytes'],
							file_name=f"genart_{idx}.png",
							mime="image/png",
							key=f"download_{idx}"  # Unique key for Streamlit
						)


if __name__ == "__main__":
	main()
