"""
Core object detection and image captioning logic.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

# Load object detection model (facebook/detr-resnet-50) 
object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
object_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load image captioning model (nlpconnect/vit-gpt2-image-captioning) -
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def detect_and_caption(pil_image):
    """
    Detect objects in a PIL image and generate a caption for the image.
    Returns (labels, caption_text).
    """
    # 1) Object detection
    inputs = object_processor(images=pil_image, return_tensors="pt")
    outputs = object_model(**inputs)
    
    # Filter out low confidence detections
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(dim=1).values > 0.9  # confidence threshold

    labels = []
    for class_idx in logits.argmax(-1)[0, keep]:
        label = object_model.config.id2label[class_idx.item()]
        labels.append(label)
    
    # 2) Image captioning
    caption_inputs = caption_processor(images=pil_image, return_tensors="pt")
    output_ids = caption_model.generate(**caption_inputs, max_length=16, num_beams=4)
    caption_text = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return labels, caption_text
