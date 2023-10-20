import cv2
from simple_facerec import SimpleFacerec

# Constants
IMAGE_DIRECTORY = "images/"
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 0, 200)
RECTANGLE_COLOR = (0, 0, 200)
RECTANGLE_THICKNESS = 4

def initialize_face_recognition(images_path):
    try:
        sfr = SimpleFacerec()
        sfr.load_encoding_images(images_path)
        return sfr
    except Exception as e:
        print(f"Error initializing face recognition: {e}")
        return None

def main():
    sfr = initialize_face_recognition(IMAGE_DIRECTORY)

    if sfr is None:
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame from the camera")
            break

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        for (y1, x2, y2, x1), name in zip(face_locations, face_names):
            cv2.putText(frame, name, (x1, y1 - 10), FONT, FONT_SCALE, FONT_COLOR, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), RECTANGLE_COLOR, RECTANGLE_THICKNESS)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




