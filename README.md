# 🧵 Fashion-MNIST CNN Klassifikator

Dieses Projekt nutzt ein **Convolutional Neural Network (CNN)** zur Klassifikation von Bildern aus dem **Fashion-MNIST-Datensatz**. Die Daten werden von Kaggle heruntergeladen, verarbeitet und für das Training eines CNN-Modells verwendet. Ziel ist es, Kleidungsstücke wie T-Shirts, Hosen oder Schuhe anhand ihrer Bilddaten automatisch zu erkennen.

---

## 📌 Projektübersicht

Dieses Projekt bietet eine **vollständige Pipeline** zur Klassifikation von Modebildern, bestehend aus:
- 📥 **Datenvorbereitung:** Laden und Vorverarbeiten der Fashion-MNIST-Bilder
- 🏗 **Modellaufbau:** Ein CNN mit mehreren Convolutional- und Pooling-Schichten
- 🎯 **Training & Optimierung:** Einsatz von **Early Stopping**, um Überanpassung zu vermeiden
- 🔎 **Fehlklassifikationsanalyse:** Anzeige falsch erkannter Bilder zur Verbesserung des Modells

Der **Fashion-MNIST-Datensatz** besteht aus **70.000** graustufigen Bildern von **10 Kleidungsstück-Kategorien**. Die Herausforderung besteht darin, aus den Bildmerkmalen Muster zu erkennen und sie den richtigen Klassen zuzuordnen.

---

## 🧠 Modellarchitektur

Ein **Convolutional Neural Network (CNN)** ist besonders gut für Bildklassifikation geeignet, da es räumliche Beziehungen in Bildern erfasst. Unser Modell besteht aus:

1. **Faltungsschichten (Convolutional Layers)** – Extrahieren Merkmale aus den Bildern.
2. **Max-Pooling-Schichten** – Reduzieren die Bildgröße, um Rechenaufwand zu sparen.
3. **Dropout-Schicht** – Verhindert Überanpassung durch zufälliges Ausschalten von Neuronen.
4. **Dense-Schichten** – Treten in der Endphase zur Klassifikation auf.

```python
tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Das Modell kann Bilder in **eine von 10 Klassen** einteilen, z. B. T-Shirt, Sneaker oder Tasche.

---

## 🏋️‍♂️ Training des Modells

Das Modell wird mit dem **Adam-Optimizer** trainiert, einem weit verbreiteten Algorithmus für neuronale Netze, der sich an die Lernrate anpasst. 

- **Loss-Funktion:** `categorical_crossentropy` – geeignet für mehrklassige Klassifikationen.
- **Batch-Größe:** 32 – Verarbeitet 32 Bilder pro Berechnungsschritt.
- **Epochen:** 10 – Das Modell sieht die gesamten Trainingsdaten 10-mal.
- **Early Stopping:** Beendet das Training, wenn sich die Validierungsgenauigkeit nicht verbessert.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels_onehot, 
          epochs=10,  
          batch_size=32, 
          validation_split=0.1, 
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
```

Falls das Modell zu früh stoppt, kann die Anzahl der Epochen erhöht werden.

---

## 📊 Ergebnisse

Nach dem Training wird das Modell auf **unabhängigen Testdaten** geprüft, um die Generalisierungsfähigkeit zu bewerten.

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot, verbose=2)
print(f"Testgenauigkeit: {test_accuracy:.2f}")
```

### 📈 Testergebnisse

✅ **Testgenauigkeit:** `0.91` (91%) – Das Modell erkennt 91 % der Bilder korrekt.  
✅ **Anzahl der Testbilder:** `10.000` – Getrennt von den Trainingsdaten.  
✅ **Fehlklassifizierte Bilder werden visualisiert** – Um mögliche Verbesserungen zu erkennen.

**Beispielhafte Konsolenausgabe:**
```
Testgenauigkeit: 0.91
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step  
Zeige 10 falsch klassifizierte Bilder...
```

Ein Wert von **91% ist gut**, aber durch Anpassungen kann die Genauigkeit weiter verbessert werden.

---

## 🖼️ Visualisierung der Vorhersagen

Ein Modell ist nur so gut wie seine Fähigkeit, Vorhersagen zu erklären. Dieses Projekt zeigt:

- **10 zufällig gewählte Bilder** mit vorhergesagten Labels.
- **Falsch klassifizierte Bilder**, um Schwächen zu erkennen.

```python
show_predictions(model, test_images, test_labels_onehot, class_names, num_images=10)
show_misclassified_images(model, test_images, test_labels_onehot, class_names, num_images=10)
```

Diese Visualisierung hilft dabei, **häufige Fehlerquellen** des Modells zu erkennen.

---

## 📦 Installation & Abhängigkeiten

### 🔽 1. Installiere die erforderlichen Bibliotheken
Damit das Skript ausgeführt werden kann, müssen einige Python-Bibliotheken installiert werden:
```bash
pip install tensorflow numpy matplotlib kagglehub
```

### 📂 2. Lade die Fashion-MNIST-Daten von Kaggle
Der Fashion-MNIST-Datensatz kann direkt über Kaggle heruntergeladen werden:
```python
import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")
```

Dies speichert die Daten lokal, sodass sie für das Training genutzt werden können.

---

## 📜 Lizenz
📃 Dieses Projekt steht unter der **MIT-Lizenz**. Weitere Details in der Datei `LICENSE`.

---

🛠️ **Erstellt von Ben Mölli**

