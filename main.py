import os
from transcriber import extract_audio, transcribe_audio
from summarizer import summarize_text
from utils import chunked_summarize

def video_to_summary(
        video_path: str, 
        model_size: str = "base", 
        summarizer_model_name: str = "facebook/bart-large-cnn",
        use_chunking: bool = False
    ) -> str:

    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path=audio_path)    
    
    transcript = transcribe_audio(audio_path, model_size=model_size)
    
    if use_chunking:
        final_summary = chunked_summarize(
            text = transcript,
            summarize_func=lambda text: summarize_text(
                text, model_name=summarizer_model_name
            ),
            max_chunk_size=2000
        )
    else:
        final_summary = summarize_text(
            transcript, 
            model_name=summarizer_model_name
        )
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return final_summary


if __name__ == "__main__":
    video_file = "What is Data Science.mp4"  # Replace with your video file path
    summary_output = video_to_summary(
        video_file, 
        model_size="base", 
        summarizer_model_name="facebook/bart-large-cnn",
        use_chunking=True,
    )

    print("Summary of the video:")
    print(summary_output)
