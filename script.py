import base64
import asyncio
import os
import wave
import struct
import datetime
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

def save_to_wav_file(audio_data, filename=None):
    """Save the audio data to a WAV file with 24KHz, 16-bit mono PCM format."""
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_output_{timestamp}.wav"
    
    # WAV file parameters for 24KHz, 16-bit mono PCM
    sample_rate = 24000
    channels = 1
    bits_per_sample = 16
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    return filename

async def text_in_audio_out():
    # Buffer to accumulate all audio data
    # Each chunk from the API is already in the proper audio format
    # We'll concatenate these chunks to form the complete audio
    audio_buffer = bytearray()
    transcript = ""
    
    async with RTLowLevelClient(
        url=endpoint,
        azure_deployment=deployment,
        key_credential=AzureKeyCredential(api_key) 
    ) as client:
        await client.send(
            ResponseCreateMessage(
                response=ResponseCreateParams(
                    modalities={"audio", "text"}, 
                    instructions="Hello."
                )
            )
        )
        done = False
        while not done:
            message = await client.recv()
            match message.type:
                case "response.done":
                    done = True
                case "error":
                    done = True
                    print(message.error)
                case "response.audio_transcript.delta":
                    transcript += message.delta
                    print(f"Transcript: {message.delta}\n", end="", flush=True)
                case "response.audio.delta":
                    # Each chunk is a base64-encoded audio segment
                    # We decode and append to our buffer
                    buffer = base64.b64decode(message.delta)
                    audio_buffer.extend(buffer)
                    print(f"Received {len(buffer)} bytes of audio data.")
                case _:
                    print(f"Received message of type: {message.type}")
    
    # Save the complete audio stream to a WAV file
    if audio_buffer:
        filename = save_to_wav_file(audio_buffer)
        print(f"\nComplete audio saved to: {filename}")
        print(f"Total audio size: {len(audio_buffer)} bytes")
    else:
        print("No audio data received.")
    
    print("\nFull transcript:")
    print(transcript)

async def main():
    await text_in_audio_out()

asyncio.run(main())