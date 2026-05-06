import streamlit as st
import cv2
import os
import time
import requests
import sys
import webbrowser
from deepface import DeepFace

# Fix Unicode print issue
sys.stdout.reconfigure(encoding='utf-8')

# ---------------- GLOBAL VARIABLE ---------------- #
last_opened_url = ""

# ---------------- FUNCTION: MESSAGE ---------------- #
def show_message(emotion):
    messages = {
        "happy": "Keep Smiling ",
        "sad": "Everything will be okay ",
        "angry": "Take a deep breath ",
        "neutral": "Stay focused ",
        "surprise": "Wow! Something exciting ",
        "fear": "Stay calm, you are strong ",
        "disgust": "Let's think positive "
    }
    return messages.get(emotion, "Have a nice day!")

# ---------------- FUNCTION: GET LOCATION ---------------- #
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()

        city = str(data.get("city", "Unknown City"))
        region = str(data.get("region", "Unknown State"))
        country = str(data.get("country", "India"))
        ip = str(data.get("ip", "Unknown IP"))
        provider = str(data.get("org", "Unknown Provider"))

        city = city.replace("ā", "a").replace("ī", "i")
        region = region.replace("ā", "a").replace("ī", "i")

        if country == "IN":
            country = "India"

        current_location = f"{city}, {region}, {country}"

        return ip, provider, current_location

    except Exception as e:
        print("Location Error:", str(e))
        return "Unknown IP", "Unknown Provider", "Location not found"

# ---------------- FUNCTION: SAVE SCREENSHOT ---------------- #
def save_screenshot(frame, emotion):
    folder = "captured_images"

    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{folder}/{emotion}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print("Screenshot saved:", filename)

# ---------------- FUNCTION: PLAY MUSIC ---------------- #
def play_music(emotion):
    global last_opened_url

    songs = {
        "happy": "https://youtu.be/jCEdTq3j-0U",
        "sad": "https://youtu.be/Qdz5n1Xe5Qo",
        "angry": "https://youtu.be/vEe-UgJvUHE",
        "neutral": "https://youtu.be/e6fg4nT653o",
        "surprise": "https://youtu.be/B9CGEsexO24",
        "fear": "https://youtu.be/BLlTFapgvOo",
        "disgust": "https://youtu.be/akjdj6iHttY"
    }

    if emotion in songs:
        new_url = songs[emotion]

        # Only open if different song
        if new_url != last_opened_url:
            print("Opening song for:", emotion)

            # Open in SAME browser tab
            webbrowser.open(new_url, new=0)

            last_opened_url = new_url

    else:
        print("No song found for:", emotion)

# ---------------- GET LOCATION ---------------- #
ip_address, internet_provider, user_location = get_location()

print("IP Address:", ip_address)
print("Internet Provider:", internet_provider)
print("Current Location:", user_location)

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# ---------------- VARIABLES ---------------- #
current_emotion = ""
current_message = ""
current_confidence = 0

# Emotion lock time
lock_duration = 10
last_change_time = 0

# ---------------- MAIN LOOP ---------------- #
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame error")
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            res = result[0]
        else:
            res = result

        # Face box
        x = res['region']['x']
        y = res['region']['y']
        w = res['region']['w']
        h = res['region']['h']

        # Emotion
        detected_emotion = res['dominant_emotion']
        confidence = res['emotion'][detected_emotion]

        now = time.time()

        # First emotion
        if current_emotion == "":
            current_emotion = detected_emotion
            current_message = show_message(current_emotion)
            current_confidence = confidence
            last_change_time = now

            save_screenshot(frame, current_emotion)
            play_music(current_emotion)

            print("First Emotion:", current_emotion)

        # Change emotion after lock duration
        elif detected_emotion != current_emotion:
            if now - last_change_time >= lock_duration:
                current_emotion = detected_emotion
                current_message = show_message(current_emotion)
                current_confidence = confidence
                last_change_time = now

                save_screenshot(frame, current_emotion)
                play_music(current_emotion)

                print("New Emotion Detected:", current_emotion)

        # ---------------- RECTANGLE ---------------- #
        color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # ---------------- EMOTION TEXT ---------------- #
        cv2.putText(
            frame,
            f"{current_emotion} ({current_confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # ---------------- MESSAGE ---------------- #
        cv2.putText(
            frame,
            current_message,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # ---------------- IP ---------------- #
        cv2.putText(
            frame,
            f"IP: {ip_address}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # ---------------- PROVIDER ---------------- #
        cv2.putText(
            frame,
            f"Provider: {internet_provider}",
            (30, 130),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (255, 0, 255),
            1
        )

        # ---------------- LOCATION ---------------- #
        cv2.putText(
            frame,
            f"Location: {user_location}",
            (30, 160),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (0, 165, 255),
            1
        )

        cv2.imshow("Mood Detector", frame)

    except Exception as e:
        print("Error:", str(e))

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()