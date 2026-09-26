# Eigenfaces, Recomendación de Artículos con NMF y Detección de Anomalías con Ensemble

**Curso:** Machine Learning
**por:** Azael Jholyem Mayta Quispe


---

## 1. Sistema de reconocimiento de rostros con Eigenfaces

**Problema:** construir un sistema de reconocimiento facial sin usar redes neuronales, usando el dataset Olivetti Faces (400 imágenes, 40 personas).

**Enfoque:**
- **PCA** sobre las imágenes aplanadas (4096 píxeles) para obtener las *eigenfaces* (direcciones de máxima varianza).
- Número de componentes elegido de forma basada en datos: el mínimo necesario para explicar el 95% de la varianza (ver notebook, Sección 1.4).
- Clasificación de la identidad sobre las proyecciones PCA usando **SVM (kernel RBF, con `GridSearchCV`)** y, como comparación, **KNN**.

**Resultados:**
- El sistema logra una exactitud alta gracias a que Olivetti Faces es un dataset controlado (rostros centrados, iluminación relativamente uniforme).
- SVM con kernel RBF superó (o igualó) a KNN al poder aprender fronteras de decisión no lineales sobre el espacio de eigenfaces.
- **Limitación central:** al ser PCA un método lineal, Eigenfaces es sensible a variaciones no controladas de iluminación, pose y expresión — no es la técnica más robusta para escenarios reales no controlados, pero es eficiente, interpretable y no requiere entrenar una red neuronal.

---

## 2. Recomendador de artículos similares con NMF + similitud de coseno

**Problema:** dado un artículo que un lector está consultando, recomendar otros artículos de temática similar (caso simulado con 20 Newsgroups, usando 6 categorías que emulan secciones de un periódico: Ciencia/Espacio, Autos, Política internacional, Salud, Tecnología, Deportes).

**Enfoque:**
- Vectorización **TF-IDF** del corpus.
- **NMF (Non-negative Matrix Factorization)** para extraer `12` tópicos latentes (matrices `W` documento-tópico y `H` tópico-término).
- Recomendación de artículos mediante **similitud de coseno** entre los vectores de tópicos (`W`) del artículo consultado y el resto del corpus.
- Evaluación mediante una **tasa de coincidencia de categoría** (proxy de relevancia) en el top-5 de recomendaciones, comparando el espacio NMF contra un baseline de TF-IDF crudo.

**Resultados:**
- Los tópicos descubiertos por NMF (sin conocer las categorías reales) suelen alinearse razonablemente con las secciones reales del corpus.
- El espacio de tópicos NMF ofrece una representación mucho más **compacta e interpretable** que el TF-IDF crudo (12 dimensiones vs. miles), a un desempeño similar o comparable en la tasa de coincidencia de categoría — una ventaja práctica relevante para un sistema de recomendación en producción.
- **Limitación central:** NMF requiere fijar de antemano el número de tópicos y es sensible al preprocesamiento del texto; al ser un modelo lineal/aditivo, no capta relaciones semánticas tan ricas como los embeddings contextuales modernos.

---

## 3. Detección de anomalías en series temporales con un modelo ensemble

**Problema:** detectar anomalías (puntuales y contextuales) en una serie temporal.

**Enfoque:**
- Serie temporal sintética (tendencia + estacionalidad + ruido) con anomalías conocidas inyectadas: picos puntuales y un cambio de nivel sostenido (anomalía contextual).
- Ingeniería de características por ventana deslizante (media móvil, desviación estándar móvil, diferencias, máximos/mínimos móviles).
- **Ensemble de 3 detectores no supervisados:** Isolation Forest (basado en aislamiento), Local Outlier Factor (basado en densidad local) y One-Class SVM (basado en frontera de decisión global).
- Combinación por **voto mayoritario** (≥2 de 3 modelos).


**Resultados:**
- El ensemble por voto mayoritario logra, en general, un F1-score igual o superior al del mejor modelo individual, reduciendo falsos positivos que cada modelo comete de forma no coincidente con los demás.
- Las anomalías puntuales (picos) fueron detectadas de forma consistente por los tres modelos; la anomalía contextual (cambio de nivel sostenido) resultó más difícil para modelos de frontera global como One-Class SVM.
- **Conclusión:** combinar detectores con distintas nociones de "anormalidad" (aislamiento, densidad, frontera) produce un sistema más robusto frente a distintos tipos de anomalías que cualquier modelo individual.

---

## Conclusión general del trabajo

Los tres ejercicios muestran que problemas hoy comúnmente resueltos con *deep learning* (reconocimiento facial, recomendación de contenido, detección de anomalías) admiten soluciones clásicas de Machine Learning — interpretables y computacionalmente económicas — cuando se combina una buena ingeniería de representación (PCA, NMF, características de ventana) con algoritmos tradicionales bien validados mediante métricas cuantitativas y comparaciones contra baselines.

## Cómo ejecutar

1. descargar el archivo y subir en Google Colab.
2. Ejecutar todas las celdas en orden (`Entorno de ejecución` → `Ejecutar todas`).
3. Los datasets (Olivetti Faces, 20 Newsgroups) se descargan automáticamente vía scikit-learn; la serie temporal de la Sección 3 es generada sintéticamente dentro del propio notebook.
