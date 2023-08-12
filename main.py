import cv2
import face_recognition
import random
import numpy as np

# Load images and encode known faces
known_face_names = ["Emma", "Laura","Abigail","Sophia","Amelia","Nora","Alice","Luna"]  # Include names for known faces
known_face_encodings = []

for name in known_face_names:
    image_path = f'Gallery/{name}.png' 
    known_image = face_recognition.load_image_file(image_path)
    known_face_locations = face_recognition.face_locations(known_image)
    
    if known_face_locations:  # Check if any face is found
        known_encoding = face_recognition.face_encodings(known_image, known_face_locations)[0]
        known_face_encodings.append(known_encoding)
    else:
        print(f"No face found in the image: {name}")

# Load the image for recognition
random_face_name = random.choice(known_face_names)
image_path = f'Gallery/{random_face_name}.png'  
unknown_image = face_recognition.load_image_file(image_path)
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
# Converting to BGR
cv2_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Loop through each face in the unknown image
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    # Compare face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    
    if any(matches):
        match_scores = [face_recognition.face_distance(known_face_encodings, face_encoding)]
        best_match_index = np.argmin(match_scores)
        name = known_face_names[best_match_index]

    # Draw rectangle and label
    cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    text_size = cv2.getTextSize(name, font, 3, 5)[0]
    cv2.rectangle(cv2_image, (left, bottom + text_size[1] + 30), (left + text_size[0] + 10, bottom), (173, 216, 230), cv2.FILLED)
    cv2.putText(cv2_image, name, (left + 6, bottom + 80), font, 3, (2, 48, 32), 5)

# Display the result
cv2.imwrite('output/recognizedFace.png', cv2_image)
