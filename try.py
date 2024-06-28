import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import numpy as np

# Path to the Tesseract executable (update this based on your installation)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Load the image using OpenCV
image_path = 'test_img.png'
image = cv2.imread(image_path)

blank_page = 255*np.ones(image.shape, dtype=np.uint8)
# blank_page = 255 * blank_page

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

confidence_color_mapping = {}
for i in range(0, 101):
    # Calculate the green component based on the confidence level
    green = int(255 * (i / 100))
    # Calculate the red component based on the confidence level
    red = 255 - green
    # Set the RGB color tuple for the confidence level
    confidence_color_mapping[i] = (red, green, 0)

# Ensure that 100 confidence level is pure green
confidence_color_mapping[100] = (0, 255, 0)

for i in range(len(results["text"])):

    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    text = results["text"][i]
    conf = int(results["conf"][i])

    if conf >= 50:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.putText(blank_page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color_mapping[conf], 0)

# cv2.imshow("Detected Image", image)
cv2.imshow("OCR Output", blank_page)
cv2.waitKey(0)
cv2.destroyAllWindows()
