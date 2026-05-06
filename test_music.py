import os

path = "music/happy.mp3"

if os.path.exists(path):
    print("File found")
else:
    print("File NOT found")