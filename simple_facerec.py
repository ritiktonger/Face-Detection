import cv2
import face_recognition
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.confidence_threshold = 0.6
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        try:
            image_files = glob.glob(os.path.join(images_path, "*.*"))
            print(f"Found {len(image_files)} encoding images.")

            for image_file in image_files:
                img = cv2.imread(image_file)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                basename = os.path.basename(image_file)
                filename, _ = os.path.splitext(basename)
                encoding = face_recognition.face_encodings(rgb_img)[0]

                self.known_face_encodings.append(encoding)
                self.known_face_names.append(filename)

            print("Encoding images loaded successfully.")
        except Exception as e:
            print(f"Error loading face encodings: {e}")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if np.any(distances < self.confidence_threshold):
                name = self.known_face_names[np.argmin(distances)]
            face_names.append(name)

        # Adjust coordinates with frame resizing
        face_locations = np.array(face_locations)
        face_locations = (face_locations / self.frame_resizing).astype(int)

        return face_locations, face_names


def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame from the camera")
            break

        face_locations, face_names = sfr.detect_known_faces(frame)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 4)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



