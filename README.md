# Análisis de Emociones en Encuesta Estudiantil

Este repositorio contiene el análisis completo de una encuesta realizada a estudiantes para detectar seis emociones (felicidad, tristeza, disgusto, ira, sorpresa y miedo) mediante técnicas de procesamiento de texto y clasificación automática.

## 📂 Estructura de carpetas
📁 **Proyecto_Mineria/**  
├─ 📁 **data/**  
│   ├─ 📁 **raw/**  
│   └─ 📁 **processed/**  
│  
├─ 📁 **notebooks/**  
│   └─ 📄 Proyecto_Minería.ipynb  
│  
├─ 📁 **src/**  
│   ├─ 📄 preprocessing.py  
│   ├─ 📄 transformers_utils.py  
│   ├─ 📄 evaluation.py  
│   └─ 📄 train_classifiers.py  
│  
│  
├─ 📁 **outputs/**  
│   ├─ 📁 **figures/**  
│   └─ 📁 **reports/**  
│  
├─ 📄 requirements.txt  
└─ 📄 README.md

## ¿Cómo reproducir los resultados?
Una de las opciones (y la más sencilla) consiste en descargar los datos dentro de la carpeta 'processed' y descargar el notebook donde esta el paso a paso de lo que se hizo y donde puedes hacer tus propios experimentos.

Para la otra opción como ejecutarlo para Visual Studio Code seguir los siguientes pasos:
1. Clona el repositorio:
   ```bash
   git clone https://github.com/DiegoCannata/Proyecto-Mineria.git
   cd Proyecto-Mineria.git

2. Crea y activa un entorno virtual, luego instala dependencias:
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # o `venv\Scripts\activate` en Windows
   pip install --upgrade pip
   pip install -r requirements.txt

3. Descarga el modelo de SpaCy:
   ```bash
   python -m spacy download es_core_news_sm

4. Descarga el notebook y los datos (de la carpeta 'raw' si los quieres en crudo y quieres hacer todo desde 0 o de la carpeta 'processed' si te quieres ir directo a los modelos)
