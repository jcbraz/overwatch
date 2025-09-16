#!/usr/bin/env python3
"""
Script to create a placeholder image for the video stream
"""
import cv2
import numpy as np

def create_placeholder_image():
    # Create a 800x600 image with gray background
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = "YOLO Video Stream"
    text2 = "Click 'Start Stream' to begin"

    # Calculate text size and position
    text1_size = cv2.getTextSize(text1, font, 2, 3)[0]
    text2_size = cv2.getTextSize(text2, font, 1, 2)[0]

    # Center the text
    text1_x = (width - text1_size[0]) // 2
    text1_y = (height - text1_size[1]) // 2 - 30

    text2_x = (width - text2_size[0]) // 2
    text2_y = (height + text2_size[1]) // 2 + 30

    # Draw text
    cv2.putText(img, text1, (text1_x, text1_y), font, 2, (255, 255, 255), 3)
    cv2.putText(img, text2, (text2_x, text2_y), font, 1, (200, 200, 200), 2)

    # Draw a border
    cv2.rectangle(img, (10, 10), (width-10, height-10), (100, 100, 100), 3)

    # Save the image
    cv2.imwrite('static/placeholder.jpg', img)
    print("Placeholder image created: static/placeholder.jpg")

if __name__ == "__main__":
    create_placeholder_image()
