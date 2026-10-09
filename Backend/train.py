import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


df = pd.read_csv("voice_dataset.csv") 

print(df.head())


X = df.drop("label", axis=1).values
y = df["label"].values


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print(label_encoder.classes_)

# =========================================
# ONE HOT ENCODING
# =========================================

y_categorical = to_categorical(y_encoded)

# =========================================
# NORMALIZAR FEATURES
# =========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42
)


model = Sequential()

# CAPA ENTRADA
model.add(Dense(
    128,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

# CAPA OCULTA 1
model.add(Dense(
    64,
    activation='relu'
))

# CAPA OCULTA 2
model.add(Dense(
    32,
    activation='relu'
))

# CAPA SALIDA
model.add(Dense(
    y_categorical.shape[1],
    activation='softmax'
))



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================================
# EVALUAR
# =========================================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nAccuracy final: {accuracy:.4f}")

# =========================================
# GUARDAR MODELO
# =========================================

model.save("models/voice_model.h5")

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nModelo guardado correctamente ")