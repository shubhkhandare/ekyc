import cv2
import numpy as np
import pytesseract
import re
import json
import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to load.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 150)
        return edged, image
    except Exception as e:
        messagebox.showerror("Error", f"Error in preprocess_image: {e}")
        return None, None

def get_document_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx
    return None

def extract_fields(ocr_text):
    name_pattern = r"Name:\s*([A-Za-z\s]+)"
    dob_pattern = r"(DOB:|Date of Birth:)\s*(\d{2}[/.-]\d{2}[/.-]\d{4})"
    doc_number_pattern = r"\b\d{10,}\b"

    name = re.search(name_pattern, ocr_text)
    dob = re.search(dob_pattern, ocr_text)
    doc_number = re.search(doc_number_pattern, ocr_text)

    return {
        'name': name.group(1).strip() if name else "Not Found",
        'date_of_birth': dob.group(2).strip() if dob else "Not Found",
        'document_number': doc_number.group(0).strip() if doc_number else "Not Found"
    }

def save_extracted_fields(extracted_fields, output_path='output/extracted_fields.json'):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as json_file:
            json.dump(extracted_fields, json_file, indent=4)
        print(f"Extracted fields saved to '{output_path}'.")
    except Exception as e:
        messagebox.showerror("Error", f"Error saving extracted fields: {e}")

def process_image(image_path):
    edged, image = preprocess_image(image_path)
    if edged is None or image is None:
        return
    
    contour = get_document_contour(edged)
    if contour is None:
        messagebox.showerror("Error", "Document contour not found.")
        return

    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    widthA = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    widthB = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
    heightB = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # OCR processing on the warped image
    text = pytesseract.image_to_string(warped)

    # Extract specific fields from the OCR text
    extracted_fields = extract_fields(text)
    
    print("\nExtracted Fields:")
    for field, value in extracted_fields.items():
        print(f"{field}: {value}")

    # Save the extracted fields to a file (JSON format)
    save_extracted_fields(extracted_fields)

    try:
        cv2.imwrite('output/extracted_document.jpg', warped)
        print("Extracted document saved to 'output/extracted_document.jpg'.")
    except Exception as e:
        messagebox.showerror("Error", f"Error saving the extracted document: {e}")

def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image",
                                            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

def create_gui():
    root = Tk()
    root.title("eKYC Document Scanner")

    Label(root, text="Upload your ID document for eKYC processing:", padx=20, pady=20).pack()

    Button(root, text="Upload Image", command=upload_image, padx=10, pady=10).pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
