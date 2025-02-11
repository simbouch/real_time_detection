import torch  # PyTorch for tensor operations and deep learning inference
import logging  # Logging module for debugging and error tracking
import numpy as np  # NumPy for handling arrays (especially OpenCV images)
import cv2  # OpenCV for image processing
from PIL import Image  # PIL (Pillow) for image handling
from transformers import DetrImageProcessor, DetrForObjectDetection  # Hugging Face DETR model components

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global references to store the model and processor to avoid redundant loading
PROCESSOR = None  # Holds the pre-trained DETR image processor
MODEL = None  # Holds the pre-trained DETR model

def load_detection_model():
    """
    Loads the DETR model and processor from Hugging Face once to avoid redundant downloads.

    The function ensures:
    - The model and processor are **only loaded once** and reused in all function calls.
    - If already loaded, it returns the existing instance.

    Returns:
        tuple: (PROCESSOR, MODEL)
            - PROCESSOR: A `DetrImageProcessor` instance for image preprocessing.
            - MODEL: A `DetrForObjectDetection` instance for object detection.
    """
    global PROCESSOR, MODEL  # Use global variables to store the model for reuse

    # If the model is already loaded, return it immediately
    if PROCESSOR is not None and MODEL is not None:
        return PROCESSOR, MODEL

    try:
        logging.info("Loading DETR model from Hugging Face...")
        # Load the processor (handles image resizing, normalization, etc.)
        PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Load the pre-trained object detection model
        MODEL = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        logging.info("Model successfully loaded.")
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise RuntimeError("Failed to load the object detection model.")  # Raise an error to avoid silent failures

    return PROCESSOR, MODEL  # Return the loaded processor and model


def convert_boxes_to_pixel_format(boxes, image_size):
    """
    Converts normalized bounding boxes from DETR to pixel-based coordinates.

    DETR outputs bounding boxes in a **normalized format**:
        - [center_x, center_y, width, height] (values between 0 and 1)

    This function converts them to **absolute pixel values**:
        - [x_min, y_min, x_max, y_max]

    Args:
        boxes (list of lists): List of bounding boxes in normalized format.
        image_size (tuple): (width, height) of the image in pixels.

    Returns:
        list of lists: Bounding boxes in pixel format (x_min, y_min, x_max, y_max).
    """
    img_w, img_h = image_size  # Extract image width and height
    pixel_boxes = []  # List to store converted bounding boxes

    for cx, cy, w, h in boxes:
        # Convert center_x, center_y, width, height -> x_min, y_min, x_max, y_max
        x_min = int((cx - w / 2) * img_w)
        y_min = int((cy - h / 2) * img_h)
        x_max = int((cx + w / 2) * img_w)
        y_max = int((cy + h / 2) * img_h)

        pixel_boxes.append([x_min, y_min, x_max, y_max])  # Append converted box

    return pixel_boxes


def detect_objects(image, threshold=0.9):
    """
    Performs object detection on an image using the DETR model.

    This function supports:
    - **PIL Images** (used by most deep learning models).
    - **OpenCV Images (NumPy arrays)** (commonly used for webcam feeds).
    - **File Paths** (images stored on disk).

    Args:
        image (PIL.Image, np.ndarray, or str): The input image for object detection.
            - If `image` is a **string**, it is treated as a file path.
            - If `image` is a **NumPy array**, it is assumed to be an OpenCV image.
            - If `image` is a **PIL Image**, it is used directly.
        threshold (float): Confidence threshold for filtering detections (default=0.9).

    Returns:
        tuple: (labels, boxes)
            - labels (list): List of detected object class names.
            - boxes (list): List of bounding boxes in pixel format (x_min, y_min, x_max, y_max).
    """
    try:
        processor, model = load_detection_model()  # Ensure the model is loaded

        # Convert the image to PIL format if needed
        if isinstance(image, str):  # If image is a file path, load it
            logging.info(f"Loading image from path: {image}")
            image = Image.open(image).convert("RGB")

        elif isinstance(image, np.ndarray):  # If image is OpenCV format, convert it to PIL
            logging.info("Converting OpenCV image to PIL format.")
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        logging.info("Processing image for object detection...")
        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Perform object detection
        outputs = model(**inputs)

        # Extract raw logits (class scores) and bounding boxes
        logits = outputs.logits  # Shape: [batch_size, num_boxes, num_classes+1]
        bboxes = outputs.pred_boxes  # Shape: [batch_size, num_boxes, 4] (center_x, center_y, width, height)

        # Convert raw logits to probabilities (excluding the background class)
        probas = logits.softmax(-1)[0, :, :-1]  # Softmax over classes
        keep = probas.max(dim=1).values > threshold  # Apply confidence threshold

        labels = []
        boxes = []
        for i in torch.where(keep)[0]:  # Iterate over valid detections
            class_id = logits.argmax(-1)[0, i].item()  # Get the highest confidence class index
            class_name = model.config.id2label[class_id]  # Get class name from model config
            box = bboxes[0, i].tolist()  # Convert tensor to list

            labels.append(class_name)  # Append class name
            boxes.append(box)  # Append bounding box

        # Convert bounding boxes from normalized (0-1) to absolute pixel format
        pixel_boxes = convert_boxes_to_pixel_format(boxes, image.size)

        logging.info(f"Detected {len(labels)} objects.")
        return labels, pixel_boxes

    except Exception as e:
        logging.error(f"Error in object detection: {e}")
        return [], []  # Return empty lists in case of failure
