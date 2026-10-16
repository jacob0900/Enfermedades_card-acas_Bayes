# Clasificación de enfermedades cardíacas con modelos bayesianos

Proyecto de Machine Learning que predice la presencia de enfermedad cardíaca a partir de 13 variables clínicas. Compara formalmente un clasificador **generativo** (Naive Bayes gaussiano) con uno **discriminativo** (regresión logística), con un pipeline libre de *data leakage*, validación cruzada estratificada y una prueba de hipótesis corregida para decidir si la diferencia entre modelos es real.

**Ver el notebook:** https://jacob0900.github.io/Enfermedades_card-acas_Bayes/

## Datos

- **Dataset:** [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland), del UCI Machine Learning Repository
- **Tamaño:** 303 pacientes y 13 variables: 5 numéricas (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) y 8 categóricas (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
- **Variable objetivo:** `num` binarizada. 0 = sin enfermedad (164 pacientes) y 1 = con enfermedad (139 pacientes, 45,9 %)
- **Valores faltantes:** 6 celdas (0,15 %), imputadas dentro del pipeline

## Metodología

1. **Diseño sin data leakage.** Partición estratificada 80/20 (242 / 61) **antes** de cualquier ajuste. Todo el preprocesamiento queda dentro de un `Pipeline` con `ColumnTransformer`:
   - Variables numéricas: imputación por mediana y `StandardScaler`
   - Variables categóricas: imputación por moda y `OneHotEncoder(drop='first')`
2. **Validación cruzada estratificada de 5 pliegues**, con los mismos pliegues para ambos modelos para poder hacer una prueba pareada. Como métrica de selección se usa **ROC-AUC**, porque no depende del umbral, tiene interpretación probabilística y es robusta al desbalance.
3. **Evaluación única en el conjunto de prueba:** matrices de confusión, curvas ROC, sensibilidad, especificidad y falsos negativos.
4. **Prueba de hipótesis** sobre la diferencia de AUC por pliegue: *t* pareado clásico y *t* corregido de **Nadeau y Bengio (2003)**, que tiene en cuenta que los pliegues de validación cruzada no son independientes.
5. **Análisis bayesiano:** interpretación de los parámetros aprendidos (`theta_`, `var_`, `class_prior_`) en unidades clínicas y discusión del supuesto de independencia condicional.
6. **Regularización de la verosimilitud** con `var_smoothing`, ajustado por `GridSearchCV` dentro de la validación cruzada.

## Resultados

| Modelo | ROC-AUC (CV) | Accuracy (test) | Sensibilidad | Especificidad | ROC-AUC (test) |
|---|---|---|---|---|---|
| Naive Bayes gaussiano | 0,862 | 0,639 | 0,893 | 0,424 | 0,820 |
| **Naive Bayes regularizado** | 0,892 | **0,902** | **0,929** | **0,879** | **0,977** |
| Regresión logística | **0,902** | 0,869 | 0,893 | 0,848 | 0,962 |

- La regresión logística supera al Naive Bayes base en los 5 pliegues, pero la prueba **corregida** da **p = 0,076**: no hay evidencia suficiente para declarar un ganador. El *t* clásico (p = 0,024) habría llevado a una conclusión equivocada.
- El Naive Bayes base **ordena bien a los pacientes, pero está mal calibrado**: el 62 % de sus probabilidades quedan saturadas (> 0,99 o < 0,01) por las varianzas casi nulas de algunas variables dummy.
- Con `var_smoothing` (ε ≈ 0,119), el accuracy en test sube de 0,64 a **0,90**, sin abandonar el supuesto *naive*. Aun así, la diferencia con la regresión logística no es significativa (p corregido = 0,504).
- Las variables numéricas más informativas son `thalach` (frecuencia cardíaca máxima) y `oldpeak` (depresión del ST). `chol` casi no discrimina.
- El supuesto de independencia condicional no se cumple en datos clínicos: por ejemplo, `age` está ligada a `thalach`, y `oldpeak`, `slope` y `exang` miden el mismo evento.

**Recomendación:** para uso clínico, usar regresión logística o Naive Bayes regularizado con probabilidades calibradas (`CalibratedClassifierCV`) y un umbral menor a 0,5 para priorizar la sensibilidad, ya que un falso negativo es mucho más costoso que un falso positivo.

## Estructura del repositorio

```
Enfermedades_card-acas_Bayes/
├── EntregableBayesiano_Resuelto.ipynb   # notebook con el análisis completo
└── docs/
    ├── index.html                       # notebook exportado (GitHub Pages)
    └── EntregableBayesiano_Resuelto.html
```

## Cómo ejecutarlo

Requisitos: Python 3.9 o superior.

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo jupyter
jupyter notebook EntregableBayesiano_Resuelto.ipynb
```

Los datos se descargan automáticamente desde el repositorio de UCI con `ucimlrepo`.

## Tecnologías

Python · scikit-learn · pandas · NumPy · SciPy · Matplotlib · Seaborn · Jupyter

## Contexto

Proyecto integrador (sección 4.4, *Modelos bayesianos*) del curso de **Machine Learning** del Dr. Lihki Rubio.

## Autor

- **Jacobo Londoño**, [LinkedIn](https://www.linkedin.com/in/jacobo-londo%C3%B1o-baquero-a2b86733a/)
