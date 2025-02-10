"""
Main entry point for the Real-Time Detection project.
"""

import gradio as gr
import logging
from detection import detect_and_caption
from utils.logger import get_logger

logger = get_logger()

def run_app():
    """
    Run the Gradio interface.
    """
    logger.info("Launching Gradio interface...")
    
    def process_image(img):
        """
        Handle incoming images from the webcam or uploader.
        """
        if img is None:
            return "No image provided", "No captions"
        objects_detected, caption_text = detect_and_caption(img)
        return str(objects_detected), caption_text

    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(source="webcam", tool="editor", type="pil"),  # or type="numpy"
        outputs=["text", "text"],
        title="Real-Time Object Detection & Image Captioning",
        description="Detect objects in real-time and generate image captions.",
        live=True
    )
    interface.launch()

if __name__ == "__main__":
    run_app()
