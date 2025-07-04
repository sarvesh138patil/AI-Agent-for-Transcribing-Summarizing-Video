import subprocess
import whisper   # openai speech to text model
import os

#subprocess executes shell commands
def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    if os.path.exists(audio_path):
        os.remove(audio_path)

    command = [
        "ffmpeg",
        "-i", video_path, #input video file
        "-q:a", "0", #quality of the audio
        "-map", "a", #map the audio stream
        audio_path, #destination path for extracted audio
        "-y"              
    ]

    subprocess.run(command, stdout= subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # Run the command and check for errors

    return audio_path

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="en")
    transcript = result["text"] 
    return transcript