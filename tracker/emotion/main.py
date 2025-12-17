import cv2
from deepface import DeepFace
from pythonosc import udp_client

# Configurazione Client OSC
# IP "127.0.0.1" per inviare allo stesso computer, porta 8000
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"Invio dati OSC su {OSC_IP}:{OSC_PORT}...")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Analisi completa delle emozioni
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # 1. Estraiamo l'emozione dominante
            dominant_emotion = results[0]['dominant_emotion']
            
            # 2. Estraiamo il valore di una specifica emozione (es. felicità 0.0 - 100.0)
            happy_score = results[0]['emotion']['happy']

            # INVIO DATI VIA OSC
            # Indirizzo /emotion per la stringa, /happy per il valore numerico
            client.send_message("/avatar/emotion", dominant_emotion)
            client.send_message("/avatar/happy", float(happy_score))

            # Visualizzazione a schermo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({happy_score:.1f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print("Analisi fallita:", e)

    cv2.imshow('Face to OSC', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()