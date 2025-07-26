# CienciaViva2025
Taller de Ciencia Viva 2025

Página web para generar marcadores:
https://chev.me/arucogen/

En la carpeta de marcadores se encuentran 5 marcadores, y el archivo del tablero para calibrar.

# Instalación
Para instalar las bibliotecas necesarias:

```bash
pip install -r requirements.tx
```
# Ejecutar códigos

Para tomar fotos y guardarlas, debes presionar la tecla s:

```bash
python3 tomarFotos.py
```

Una vez que se terminen de tomar las fotos, para calibrar la cámara
ese necesario ejectutar e ir dando enter, hasta que se muestre la matriz 
de calibración en la terminal:

```bash
python3 calibracionCamara.py
```

Para la detección del marcador Aruco es necesario que ya se encuentre calibrada
la cámara:

```bash
python3 deteccionAruco.py
```

Para la detección de gestos:

```bash
python3 deteccionGestos.py
```

Para el programa de realidad aumentada:

```bash
python3 gestosRA.py
```