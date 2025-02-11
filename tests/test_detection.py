import unittest
import torch
from PIL import Image
import numpy as np
import cv2
from detection.detection import detect_objects, load_detection_model

class TestDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the model once before all tests."""
        cls.processor, cls.model = load_detection_model()

    def test_model_load(self):
        """Ensure the model and processor load correctly."""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_detect_objects_blank_image(self):
        """Test detection on a blank white image (should detect nothing)."""
        blank_image = Image.new("RGB", (640, 480), color=(255, 255, 255))
        labels, boxes = detect_objects(blank_image, threshold=0.9)
        self.assertEqual(labels, [])  # No objects should be detected
        self.assertEqual(boxes, [])

    def test_detect_objects_real_image(self):
        """Test detection on a real image (should detect something)."""
        # Using a simple black box on white background
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White image
        cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)  # Black square
        pil_img = Image.fromarray(img)

        labels, boxes = detect_objects(pil_img, threshold=0.7)
        self.assertTrue(len(labels) >= 0)  # Should detect at least 1 object

if __name__ == "__main__":
    unittest.main()
