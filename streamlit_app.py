import streamlit as st
from openai import OpenAI

# --- App Title and Description ---
# Sets the title and a brief description of the Streamlit app.
st.title("📄 Document Question Answering")
st.write(
    "Upload a document (.txt or .md) and ask a question. The app will use an AI model to find the answer within the document."
    " You'll need an OpenAI API key to use this app. You can get one from the [OpenAI Platform](https://platform.openai.com/account/api-keys)."
)

# --- API Key Configuration ---
# Tries to get the OpenAI API key from Streamlit's secrets management.
# This is a secure way to store sensitive information.
try:
    openai_api_key = st.secrets["API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "ERROR: OpenAI API key not found. Please add it to your secrets file.", icon="🚨")
    st.stop()


# --- Main Application Logic ---
# This block only runs if the API key is available.
if not openai_api_key:
    st.info(
        "Please add your OpenAI API key to your secrets file to continue.", icon="🗝️")
else:
    # Create an OpenAI client instance with the provided API key.
    client = OpenAI(api_key=openai_api_key)

    # --- File Uploader ---
    # Allows the user to upload a document.
    # The app will accept .txt and .md files.
    uploaded_file = st.file_uploader(
        "Upload your document", type=("txt", "md")
    )

    # --- Question Input ---
    # A text area for the user to ask their question.
    # This input is disabled until a file is uploaded to guide the user.
    question = st.text_area(
        "Ask a question about the document",
        placeholder="e.g., Can you give me a short summary of the document?",
        disabled=not uploaded_file,
        height=100,
    )

    # --- Generate and Display Answer ---
    # This block runs only when a file has been uploaded and a question has been asked.
    if uploaded_file and question:
        try:
            # Read the content of the uploaded file.
            document = uploaded_file.read().decode()

            # Construct the prompt for the AI model.
            # It includes the document content and the user's question.
            messages = [
                {
                    "role": "user",
                    "content": f"You are an expert at answering questions based on a provided document.\n\n"
                    f"Here is the document:\n\n---\n\n{document}\n\n---\n\n"
                    f"Based on the document, please answer the following question: {question}",
                }
            ]

            # Call the OpenAI API to generate a response.
            # We use stream=True to get the response chunk by chunk for a better user experience.
            with st.spinner("Generating answer..."):
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    stream=True,
                )

                # Use st.write_stream to display the streaming response in the app.
                st.write_stream(stream)

        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")
