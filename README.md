> [Ver en ingles/See in english](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/README.md)
# 🤗 Entrenador de Modelos NER Transformers
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![image](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)](https://code.visualstudio.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Entrena tu propio modelo NER usando Transformers de HuggingFace con estos Scripts.

## 📝 Explicación de Tecnología
Un Modelo NER (Named Entity Recognition o Reconocimiento de Entidades Nombradas) es una herramienta de IA capaz de reconocer palabras y patrones y clasificarlos, dependiento de los datos con los que haya sido entrenado.\
Existen modelos ya entrenados como  [SpaCy](https://spacy.io/) pero con este simple Script puedes entrenar tus propios modelos con sets de datos personalizados.

## 🛠️ Instalación
Obviamente necesitarás Python para instalar las dependencias y ejecutar los scripts.\
Abre un CMD e instala las dependencias necesarias:
```
pip install transformers datasets seqeval scikit-learn torch transformers[torch] accelerate>=0.26.0
```
O si te falla o tienes una versión más nueva de Python:
```
py -m pip install transformers datasets seqeval scikit-learn torch transformers[torch] accelerate>=0.26.0
```
Verifica que Python esté en el PATH de Windows
`C:\Users\USER_NAME\AppData\Local\Programs\Python\Python313\Scripts`

> [!NOTE]
> La carpeta `Python313\` representa que la versión instalada es la '3.13', si tienes otra versión, cambialo.

## 🚀 Explicación de uso del proyecto
En [trainmodel.py](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/trainmodel.py), cambia el set de datos por defecto con los datos con los que quieres entrenar tu modelo NER (explicado en los comentarios).\
En la `linea 77` están los argumentos de entrenamiento, puedes cambiarlos para probar como salen los resultados.\
Ejecuta el script de entrenamiento con:
```
python trainmodel.py
```
O si te falla o tienes una versión más nueva de Python:
```
py trainmodel.py
```
Ahora, en [inferencemodel.py](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/inferencemodel.py), cambia el mapa de etiquetas para que coincida con el usado en el entrenamiento.\
Ejecuta el script de inferencia:
```
python inferencemodel.py
```
O si te falla o tienes una versión más nueva de Python:
```
py inferencemodel.py
```

## 📂 Archivos
Si los scripts se ejecutan correctamente, el modelo se generará en la misma carpeta, estos ficheros son:
- **my_ner_model**: aqui se guarda toda la información y configuración del modelo.
- **ner_model**: aqui se guardan los diferentes Checkpoints del modelo.

## 💻 Tecnologías usadas
- Lenguaje de programación: [Python](https://www.python.org/)
- Framework: [seqeval](https://github.com/chakki-works/seqeval) (1.2.2)
- Librerias:
  - [datasets](https://pypi.org/project/datasets/) (3.3.2)
  - [scikit-learn](https://pypi.org/project/scikit-learn/) (1.6.1)
  - [torch](https://pypi.org/project/torch/) (2.6.0)
  - [transformers (con PyTorch)](https://huggingface.co/docs/transformers/en/installation)
  - [accelerate](https://pypi.org/project/accelerate/) (0.26.0)
- IDE Recomendado: [VS Code](https://code.visualstudio.com/)
