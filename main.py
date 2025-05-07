import cv2
import time
import threading
from face_detection import detect_faces
from weapon_detection import detect_weapons
from alarm import start_alarm, stop_alarm
from utils import load_known_faces
from email_sender import send_email_with_attachment

CONFIDENCE_THRESHOLD = 0.6
ALARM_COOLDOWN = 5  
FRAME_SKIP = 1 

def capture_and_send_email(frame):
    """Captures a screenshot and sends an email alert."""
    image_path = "screenshot.png"
    cv2.imwrite(image_path, frame)
    print("Screenshot captured. Sending email...")
    send_email_with_attachment(image_path)

def detect_objects_in_realtime():
    cap = cv2.VideoCapture(0)
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_alarm_time = 0
    alarm_playing = False
    alarm_count = 0
    frame_count = 0

    known_faces = load_known_faces("facenet_embeddings1.npy")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  

        result = {}

        
        face_thread = threading.Thread(target=lambda: result.update({"face": detect_faces(frame, known_faces, CONFIDENCE_THRESHOLD)}))
        weapon_thread = threading.Thread(target=lambda: result.update({"weapon": detect_weapons(frame)}))

        face_thread.start()
        weapon_thread.start()
        face_thread.join()
        weapon_thread.join()

        unknown_detected = result.get("face", False)
        weapon_detected = result.get("weapon", False)

        print(f"Unknown: {unknown_detected}, Weapon: {weapon_detected}")

        if unknown_detected and weapon_detected:
            current_time = time.time()

            if not alarm_playing or (current_time - last_alarm_time > ALARM_COOLDOWN):
                print("🚨 Starting alarm!")
                threading.Thread(target=start_alarm).start()
                alarm_playing = True
                last_alarm_time = current_time
                alarm_count += 1

                if alarm_count > 2:
                    print("📸 Taking screenshot and sending email...")
                    threading.Thread(target=capture_and_send_email, args=(frame,)).start()

        elif alarm_playing and (not unknown_detected or not weapon_detected):
            print("🛑 Stopping alarm!")
            stop_alarm()
            alarm_playing = False

        cv2.imshow("🔍 Real-time Face & Weapon Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()  

if __name__ == "__main__":
    detect_objects_in_realtime()
