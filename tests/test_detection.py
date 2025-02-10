"""
Basic unit tests for the detection module.
"""

import unittest
from PIL import Image
from detection import detect_and_caption

class TestDetection(unittest.TestCase):
    def test_detect_and_caption_empty_image(self):
        """
        Test detect_and_caption with a simple blank image.
        Ensures it returns valid results without throwing errors.
        """
        blank_image = Image.new("RGB", (224, 224), "white")
        objects, caption = detect_and_caption(blank_image)
        
        self.assertIsInstance(objects, list)
        self.assertIsInstance(caption, str)
        # You could further test for expected values, etc.

if __name__ == "__main__":
    unittest.main()
