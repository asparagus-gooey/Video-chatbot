import os
import cv2
import numpy as np
from PIL import Image 
from glob import glob
from huggingface_hub import login
import face_recognition as face_reg
from transformers import AutoTokenizer, AutoModelForCausalLM

# Lists to store known face encodings and names
known_face_encodings = []
known_face_names = []


def LoadKnownFace(face_path):
    # Use glob to find all jpg images in the specified directory
    faces = glob(face_path + "/**.jpg", recursive=True)

    # Check if any images were found
    if not faces:
        print(f"No images found at path: {face_path}")
        return

    for face_image_path in faces:
        # Load the image file
        pil_image = Image.open(face_image_path).convert("RGB")  
       
        image = np.array(pil_image)  
    
        encoded_face = face_reg.face_encodings(image)
        
        if encoded_face:
            known_face_encodings.append(encoded_face[0])  # Take the first encoding found
            # Use the filename (without extension) as the name
            known_face_names.append(os.path.basename(face_image_path).split('.')[0])
        else:
            print(f"No face found in image: {face_image_path}")


LoadKnownFace("known_faces")


def compare_with_database(face_encoding):
    matches = face_reg.compare_faces(known_face_encodings, face_encoding)
    return any(matches)

# Facial recognition authentication
def authenticate_user():
    # Initialize webcam capture for facial recognition
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return False

    ret, frame = video_capture.read()
    video_capture.release()
    cv2.destroyAllWindows()

    if not ret:
        print("Error: Failed to capture image from camera.")
        return False

    # Convert frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ensure the frame is in the correct format
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        print("Error: Frame is not in the correct RGB format.")
        return False

    face_encodings = face_reg.face_encodings(frame)

    if face_encodings:
        face_encoding = face_encodings[0]
        matches = compare_with_database(face_encoding)
        if matches:
            return True  # Proceed to chatbot
    return False

def chatbot_interaction():
    print("Chatbot: Hello! Type 'end conversation' to close the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "end conversation":
            print("Chatbot: Ending conversation. Returning to facial recognition.")
            break

        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=1024)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("Chatbot:", response)

# Main loop to alternate between facial recognition and chatbot
def main():
    while True:
        print("Please look at the camera for authentication.")
        if authenticate_user():
            print("Authentication successful. Starting chatbot conversation.")
            chatbot_interaction()
        else:
            print("Authentication failed. Please try again.")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./llm_model")
    model = AutoModelForCausalLM.from_pretrained("./llm_model")
    print("Model Successfully Downloaded\n\n")
    print("Running application")
    main()
    
