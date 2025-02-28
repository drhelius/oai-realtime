import base64
import asyncio
import os
import wave
import datetime
import tempfile
import streamlit as st
import sounddevice as sd
import numpy as np
from azure.core.credentials import AzureKeyCredential
from rtclient import (
    ResponseCreateMessage,
    RTLowLevelClient,
    ResponseCreateParams
)
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["AZURE_OPENAI_API_KEY"]    
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

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

async def generate_response(prompt):
    """Generate a text and audio response from the LLM based on the user prompt."""
    text_placeholder = st.empty()

    async with RTLowLevelClient(
        url=endpoint,
        azure_deployment=deployment,
        key_credential=AzureKeyCredential(api_key) 
    ) as client:
        await client.send(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                    modalities={"audio", "text"}, 
                    instructions=prompt
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

def create_ui():
    """Create the Streamlit user interface."""
    st.set_page_config(page_title="AI Speech Generation", page_icon="🎙️")

    st.title("🎙️ AI Speech Generator")
    st.markdown("""
    Enter your prompt below and the AI will generate a spoken response.
    """)

    user_input = st.text_area("Enter your prompt:", height=100, 
                             placeholder="e.g., Please explain quantum computing in simple terms.")

    return user_input

def app():
    """Main Streamlit application."""
    user_input = create_ui()

    if st.button("Generate Response"):
        if not user_input:
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating response..."):
            transcript, audio_file = asyncio.run(generate_response(user_input))

        st.success("Response generated!")

        st.subheader("Text Response")
        st.markdown(transcript)

        display_audio_response(audio_file)

if __name__ == "__main__":
    app()
