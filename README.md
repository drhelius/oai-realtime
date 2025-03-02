# OpenAI Real-time Audio Generation

This repository contains tools for generating real-time text and audio responses using Azure OpenAI's real-time capabilities. It demonstrates how to interact with Azure OpenAI's API to convert text prompts into both text and synthesized speech.

## Features

- Real-time text-to-speech generation using Azure OpenAI
- Text transcript alongside audio output
- Command-line interface (script.py)
- Interactive Streamlit web interface (app.py)
- Audio playback and download capabilities in the web interface

## Prerequisites

- Python 3.10 or higher
- Azure OpenAI API access with appropriate credentials
- Audio playback capability for testing

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/oai-realtime.git
   cd oai-realtime
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install "https://github.com/Azure-Samples/aoai-realtime-audio-sdk/releases/download/py%2Fv0.5.3/rtclient-0.5.3.tar.gz"
   ```

3. Set up your environment variables by creating a `.env` file with:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   OPENAI_API_VERSION=2024-10-21
   ```

## Usage

### Command Line Script

Run the basic script for a simple demo:

```bash
python script.py
```

This will:
- Generate a response to a predefined prompt
- Save the audio output as a WAV file
- Display the text transcript

### Streamlit Web Interface

For an interactive experience, run:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Enter custom prompts
- See real-time text responses
- Listen to AI-generated audio responses
- Download the audio files
