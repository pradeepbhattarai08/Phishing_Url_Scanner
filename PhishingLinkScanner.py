import streamlit as st
import openai
import validators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API Key (Replace with your actual API key)
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Function to analyze URL with OpenAI
def analyze_url(url):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo as an alternative model
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert helping to detect phishing URLs."},
                {"role": "user", "content": f"Analyze this URL and determine if it looks like a phishing attempt: {url}"}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        # Check if the error is related to quota exceeded
        if 'insufficient_quota' in str(e):
            logger.error("Quota exceeded. Please check your OpenAI API plan.")
            return "Error: You have exceeded your API quota. Please check your plan."
        else:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error analyzing the URL: {e}"

# Function to handle the Streamlit app logic
def run_phishing_scanner():
    # Streamlit UI
    st.title("Phishing Link Scanner")
    st.write("Enter a URL to check if it's phishing or safe.")

    url = st.text_input("Enter URL:")
    if st.button("Scan"):
        if not url:
            st.error("Please enter a URL.")
        elif not validators.url(url):
            st.error("Invalid URL format. Please enter a valid URL.")
        else:
            logger.info(f"Received URL for scanning: {url}")
            result = analyze_url(url)
            
            if "phishing" in result.lower():
                st.error(f"⚠️ Potential Phishing Detected! Analysis: {result}")
            else:
                st.success(f"✅ Safe link. Analysis: {result}")

# Main function to run the app
def main():
    run_phishing_scanner()

# Entry point for the app
if __name__ == "__main__":
    main()