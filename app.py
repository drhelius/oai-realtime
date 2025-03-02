# Standard library imports
import asyncio
import base64
import datetime
import os
import tempfile
import wave

# Third-party imports
import numpy as np
import sounddevice as sd
import app as st
from azure.core.credentials import AzureKeyCredential

from rtclient import (
    ResponseCreateMessage,
    RTLowLevelClient,
    ResponseCreateParams
)

# Local imports
import models_config

# Constants
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = np.int16

def setup_audio_stream():
    """Initialize and return an audio output stream."""
    audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16'
    )
    audio_stream.start()
    return audio_stream

def play_audio_chunk(audio_stream, buffer):
    """Process and play an audio chunk through the given stream."""
    audio_chunk = np.frombuffer(buffer, dtype=np.int16)
    audio_stream.write(audio_chunk)

def save_to_wav_file(audio_data, filename=None):
    """Save the audio data to a WAV file with 24KHz, 16-bit mono PCM format."""
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_output_{timestamp}.wav"

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data)

    return filename

async def process_llm_response(client, text_placeholder):
    """Process the LLM response and handle different message types."""
    audio_buffer = bytearray()
    transcript = ""
    audio_stream = setup_audio_stream()

    try:
        done = False
        while not done:
            message = await client.recv()
            match message.type:
                case "response.done":
                    done = True
                case "error":
                    done = True
                    st.error(f"Error: {message.error}")
                case "response.audio_transcript.delta":
                    transcript += message.delta
                    text_placeholder.markdown(transcript)
                case "response.audio.delta":
                    buffer = base64.b64decode(message.delta)
                    play_audio_chunk(audio_stream, buffer)
                    audio_buffer.extend(buffer)
                case _:
                    pass
    finally:
        audio_stream.stop()
        audio_stream.close()

    return transcript, audio_buffer

def create_audio_file(audio_buffer):
    """Create a temporary audio file from the buffer and return its path."""
    if not audio_buffer:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_filename = tmp_file.name

    return save_to_wav_file(audio_buffer, temp_filename)

async def generate_response(prompt, model_id):
    """Generate a text and audio response from the LLM based on the user prompt."""
    text_placeholder = st.empty()
    
    try:
        env_keys = models_config.get_env_variable_keys(model_id)
        endpoint = os.environ[env_keys["endpoint"]]
        api_key = os.environ[env_keys["api_key"]]
        deployment = os.environ[env_keys["deployment_name"]]
    except (ValueError, KeyError) as e:
        st.error(f"Model configuration error: {e}. Please check models in .env file.")
        return None, None

    async with RTLowLevelClient(
        url=endpoint,
        azure_deployment=deployment,
        key_credential=AzureKeyCredential(api_key)
    ) as client:
        await client.send(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                    modalities={"audio", "text"},
                    instructions=prompt,                    
                )
            )
        )

        transcript, audio_buffer = await process_llm_response(client, text_placeholder)

    audio_file = create_audio_file(audio_buffer)
    return transcript, audio_file

def display_audio_response(audio_file):
    """Display audio response and provide download option."""
    if not audio_file:
        return

    st.subheader("Audio Response")

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    st.audio(audio_bytes, format="audio/wav")

    st.download_button(
        label="Download Audio",
        data=audio_bytes,
        file_name="ai_response.wav",
        mime="audio/wav"
    )

    try:
        os.unlink(audio_file)
    except Exception as e:
        st.warning(f"Could not remove temporary file: {e}")

def create_sidebar():
    """Create the sidebar with model selection."""
    st.sidebar.title("Model Settings")
    
    model_options = models_config.get_model_names()
    
    if not model_options:
        st.sidebar.error("No models configured. Please add model configurations to your .env file.")
        st.stop()  # Stop the app if no models are configured
    
    # Display as (value, label) pairs
    model_dict = dict(model_options)
    model_names = list(model_dict.values())
    selected_name = st.sidebar.selectbox("Select Model", model_names)
    
    # Find the model_id that corresponds to the selected name
    model_id = next(k for k, v in model_dict.items() if v == selected_name)
    
    return model_id

def create_ui():
    """Create the Streamlit user interface."""
    st.title("🎙️ AI Speech Generator")
    st.markdown("""
    Enter your prompt below and the AI will generate a spoken response.
    """)

    user_input = st.text_area("Enter your prompt:", height=100, 
                             placeholder="e.g., Please explain quantum computing in simple terms.")

    return user_input

def app():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="AI Speech Generation",
        page_icon="🎙️",
        layout="wide",
    )

    model_id = create_sidebar()
    user_input = create_ui()

    if st.button("Generate Response"):
        if not user_input:
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating response..."):
            transcript, audio_file = asyncio.run(generate_response(user_input, model_id))
            
        if transcript is None:
            st.error("Failed to generate response. Please check model configuration.")
            return

        st.success("Response generated!")

        st.subheader("Text Response")
        st.markdown(transcript)

        display_audio_response(audio_file)

if __name__ == "__main__":
    app()
