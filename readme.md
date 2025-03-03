Handschriftenerkennung mit CNN

Dieses Projekt implementiert ein Convolutional Neural Network (CNN) zur Erkennung handgeschriebener Buchstaben (A-Z). Es nutzt TensorFlow/Keras zur Modellerstellung, OpenCV für Bildverarbeitung und Tkinter für eine Benutzeroberfläche zur Handschrifterkennung.

Installation

1. Voraussetzung

Das Projekt wurde mit Python 3.12.7 getestet. Stelle sicher, dass du diese oder eine kompatible Version installiert hast.

2. Klone das Repository

3. Installiere die Abhängigkeiten

Nutze pip, um die benötigten Pakete zu installieren:

pip install -r requirements.txt

Falls Fehler auftreten, stelle sicher, dass du eine virtuelle Umgebung nutzt:

python -m venv venv
source venv/bin/activate  # Für Linux/macOS
venv\Scripts\activate  # Für Windows
pip install -r requirements.txt

Nutzung

1. Daten vorbereiten

Die Trainingsbilder sollten in einem Ordner mit einzelnen Buchstaben als Unterverzeichnisse abgelegt sein:

Letters/
├── A/
│   ├── img1.png
│   ├── img2.png
├── B/
│   ├── img1.png
│   ├── img2.png
...

2. Modell trainieren oder laden

Führe das Skript compile_model.py aus, um das Modell zu trainieren oder ein bestehendes Modell zu laden:

python compile_model.py

Das trainierte Modell wird als model_kisy.h5 gespeichert.

3. Handschrifterkennung ausführen

Führe predict.py aus, um die grafische Benutzeroberfläche für Handschrifterkennung zu starten:

python predict.py

Eine Zeichenfläche öffnet sich, auf der du Buchstaben zeichnen kannst. Das Modell erkennt den Buchstaben und zeigt die Vorhersage an.

Projektstruktur

📂 dein-repo/
├── 📄 compile_model.py  # Trainiert oder lädt das Modell
├── 📄 get_data.py       # Lädt Bilddaten und Labels
├── 📄 predict.py        # Startet die GUI zur Handschrifterkennung
├── 📄 requirements.txt  # Liste der benötigten Pakete
├── 📄 README.md         # Diese Dokumentation




