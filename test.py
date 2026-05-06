from keras.models import load_model

model = load_model('model/emotion_model.hdf5')
print("Model loaded successfully")