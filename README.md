# Laboratorio 1: Regresión Lineal

## Información del proyecto

- **Curso:** Machine Learning Labs
- **Laboratorio:** 01 - Linear Regression
- **Lenguaje:** Python 3
- **Notebook principal:** `linear_regression.ipynb`
- **Dataset principal:** California Housing
- **Dataset de aprendizaje en línea:** Year Prediction MSD

## Entorno de ejecución

El notebook se ejecuta con Python 3 y utiliza las siguientes librerías:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Antes de ejecutar el notebook, se deben ejecutar las celdas de configuración e imports. En Google Colab, la primera celda instala o actualiza las dependencias necesarias.

## 1. Preparación y configuración

El notebook establece una semilla aleatoria (`RANDOM_STATE = 42`) para que los resultados sean reproducibles. También define el tamaño del conjunto de prueba, el número de folds para validación cruzada y la rejilla de valores de regularización `alpha`.

## 2. Análisis exploratorio del California Housing

El conjunto California Housing contiene variables relacionadas con ingresos, edad de las viviendas, habitaciones, población, ocupación y ubicación geográfica.

### Observaciones

- `MedInc` es la variable con mayor relación con el valor medio de la vivienda.
- La variable objetivo presenta un límite superior de 5.0, equivalente a $500,000.
- Los valores iguales a 5.0 se consideran observaciones censuradas y se excluyen del modelado.
- Algunas variables presentan asimetría positiva, por lo que se aplica `log1p` a las variables seleccionadas.

## 3. Ridge Regression y pipelines

El pipeline utiliza las siguientes etapas:

1. Transformación logarítmica de variables sesgadas.
2. Expansión de características polinomiales.
3. Estandarización con `StandardScaler`.
4. Regresión Ridge.
5. Búsqueda del mejor `alpha` mediante `GridSearchCV`.

El uso de un pipeline evita fuga de información porque cada transformación se ajusta únicamente con los datos de entrenamiento de cada fold.

## 4. Ejercicio 1.1: efecto del grado polinomial

Se compararon modelos Ridge con grados polinomiales 1, 2 y 3.

| Grado | Mejor alpha | CV RMSE | Test RMSE |
|---:|---:|---:|---:|
| 1 | 2.5929 | 0.6183 | 0.6285 |
| 2 | 0.0024 | 0.5452 | 0.5513 |
| 3 | 0.0574 | 0.5336 | 0.5391 |

### Interpretación

El modelo de grado 3 obtuvo el menor `Test RMSE`, con un valor de 0.5391, equivalente aproximadamente a $53,910. En este experimento, aumentar el grado mejoró el resultado porque permitió representar relaciones no lineales más complejas. Sin embargo, un grado mayor no siempre mejora el rendimiento: puede aumentar la varianza y producir sobreajuste. Por eso es necesario utilizar validación cruzada y regularización.

## 5. Ejercicio 1.2: Ridge vs. Lasso

### Objetivo

Comparar Ridge y Lasso utilizando la misma transformación polinomial de grado 2.

### Resultados

Resultados obtenidos ejecutando la celda en Python 3:

| Modelo | Mejor alpha | Test RMSE | Coeficientes iguales a cero |
|---|---:|---:|---:|
| Ridge | 0.0024 | 0.5513 | 0 de 44 |
| Lasso | 0.0001 | 0.5657 | 7 de 44 |

### Interpretación de los coeficientes

- Ridge reduce el tamaño de los coeficientes, pero normalmente no los convierte exactamente en cero.
- Lasso puede convertir coeficientes exactamente en cero debido a la penalización L1.
- Los coeficientes cero indican que Lasso está realizando selección de características.
- La gráfica `assets/lasso_top20_coefficients.png` muestra las 20 características polinomiales con mayor impacto absoluto.

### Respuestas del ejercicio 1.2

1. **Ridge obtuvo el mejor Test RMSE**, con 0.5513, mientras Lasso obtuvo 0.5657.
2. Lasso anuló 7 de 44 coeficientes y Ridge no anuló ninguno. Esto confirma que Lasso realiza selección de características mediante la penalización L1.
3. Ridge conserva las características, aunque reduce sus pesos; Lasso genera un modelo más disperso y potencialmente más interpretable.
4. La gráfica `assets/lasso_top20_coefficients.png` muestra las 20 características polinomiales con mayor impacto absoluto. Las barras rojas representan coeficientes negativos y las verdes coeficientes positivos.

### Optimización de la ejecución

Para reducir el tiempo sin eliminar ninguna tarea se utilizaron `n_jobs=-1`, tres folds para Lasso, una rejilla de 12 valores de `alpha` y `max_iter=5_000`. Se mantienen `GridSearchCV`, la comparación Ridge/Lasso, el `Test RMSE`, el conteo de ceros y la gráfica solicitada.

## 6. Evaluación del modelo

El desempeño se analiza mediante:

- RMSE de validación cruzada.
- RMSE sobre el conjunto de prueba.
- Error porcentual absoluto.
- Gráficas de residuos.
- Error según rangos de precio.

El RMSE está expresado en unidades de $100,000. Para convertirlo aproximadamente a dólares, se multiplica por 100,000.

## 7. Aprendizaje en línea con SGDRegressor

La segunda parte utiliza el dataset Year Prediction MSD. Debido a su tamaño, los datos se procesan por bloques mediante `partial_fit`.

### Principios utilizados

- El escalador se ajusta únicamente con los datos de entrenamiento.
- Los bloques se transforman antes de actualizar el modelo.
- `partial_fit` actualiza los pesos sin reiniciar el modelo.
- El RMSE de validación se calcula sobre el conjunto de prueba.

## 8. Ejercicio 2.1: entrenamiento por mini-lotes

### Acciones realizadas

1. Se creó `StandardScaler` y se ajustó con `partial_fit` sobre cada chunk de entrenamiento.
2. Se creó `SGDRegressor` con `loss='squared_error'`, `learning_rate='constant'`, `eta0=0.001` y `random_state=42`.
3. Se recorrieron nuevamente los chunks, se escalaron las características y se llamó a `sgd.partial_fit(X_scaled, y_chunk)`.
4. Después de cada actualización se calculó y guardó el RMSE en `chunk_rmses`.

### Resultados obtenidos en Python 3

- Número de chunks usados para ajustar el scaler: **93**
- Número de chunks usados para entrenar: **93**
- RMSE del primer chunk: **44.2414 años**
- RMSE del último chunk: **9.7518 años**

### Respuestas al enunciado

1. El modelo fue creado con la configuración solicitada: error cuadrático, tasa constante `eta0=0.001` y semilla 42.
2. El scaler se ajustó únicamente con los datos de entrenamiento y se reutilizó para transformar cada chunk.
3. `partial_fit` permitió actualizar el modelo progresivamente sin cargar las 463,715 filas completas en memoria.
4. El RMSE disminuyó considerablemente entre el primer y el último chunk. Esto indica aprendizaje durante la pasada, aunque el error puede variar porque cada chunk tiene una distribución diferente de canciones.

> Nota: con `MSD_TRAIN_ROWS = 463,715` y `CHUNK_SIZE = 5,000`, el número correcto de chunks es 93. Esto explica la diferencia con la estimación aproximada de 10–11 chunks escrita originalmente en el enunciado.

## 9. Ejercicio 2.2: curva de convergencia

### Acciones realizadas

1. Se reutilizó la lista `chunk_rmses` generada en el ejercicio 2.1.
2. Se creó una gráfica de RMSE en función del número de chunk.
3. Se identificaron el RMSE inicial, el RMSE final y el RMSE mínimo.
4. La gráfica se guardó en `assets/sgd_convergence_curve.png`. Si se vuelve a ejecutar la celda, se crea una nueva versión para conservar la anterior.

### Resultados obtenidos

- RMSE del primer chunk: **44.2414 años**
- RMSE del último chunk: **9.7518 años**
- RMSE mínimo: **7.7227 años**
- Chunk donde se alcanzó el mínimo: **53**

### Respuestas al enunciado

1. El RMSE disminuye de 44.2414 a 9.7518 años, por lo que el modelo muestra una mejora general durante una sola pasada.
2. La curva no es perfectamente descendente: presenta oscilaciones y algunos picos. Esto es normal porque cada chunk contiene canciones con una distribución diferente.
3. El valor mínimo se alcanzó en el chunk 53, con un RMSE de 7.7227 años.
4. El modelo aprende durante la primera época, aunque las variaciones entre chunks indican que una sola métrica no debe interpretarse de forma aislada.

## 10. Ejercicio 2.3: entrenamiento multi-época

### Resultados

| Época | Validation RMSE |
|---:|---:|
| 1 | 9.9486 |
| 2 | 9.9922 |
| 3 | 10.3767 |
| 4 | 9.9292 |
| 5 | 10.0102 |
| 6 | 9.9112 |
| 7 | 9.8602 |
| 8 | 10.0969 |
| 9 | 10.0922 |
| 10 | 9.8289 |

### Acciones realizadas

1. Se cargó el conjunto de prueba oficial y se transformó con el scaler ajustado en el ejercicio 2.1.
2. Se prepararon los 93 chunks escalados una sola vez para evitar leer y transformar el archivo en cada época.
3. Se creó un `SGDRegressor` nuevo con la configuración solicitada.
4. En cada época se barajó el orden de los chunks y se actualizó el modelo con `partial_fit`.
5. Se registró el `train RMSE` por chunk y el `validation RMSE` por época.
6. La gráfica se guardó en `assets/sgd_multiepoch_convergence.png` sin sobrescribir versiones anteriores.

### Respuestas al enunciado

1. La `validation RMSE` inicial fue **9.9486** y la final fue **9.8289** años.
2. La validación mejora en general, aunque presenta fluctuaciones entre épocas.
3. El mejor resultado se alcanzó en la **época 10**, con `validation RMSE = 9.8289`.
4. La curva muestra cierta saturación: las mejoras son pequeñas y el error oscila aproximadamente alrededor de 10 años. Entrenar más épocas no garantiza una mejora significativa y podría producir sobreajuste.

### Interpretación

El número de épocas debe seleccionarse usando el menor error de validación, no únicamente el error de entrenamiento. En este experimento, la décima época fue la mejor, pero la diferencia respecto a la primera fue pequeña, por lo que el beneficio de continuar entrenando es limitado.

## 11. Ejercicio 2.4: exploración de la tasa de aprendizaje

Se comparan las tasas:

`0.0001`, `0.001`, `0.01`, `0.1` y `1.0`.

### Resultados obtenidos

| eta0 | Estado | Chunks procesados | RMSE final |
|---:|---|---:|---:|
| 0.0001 | Estable | 93 | 9.4396 |
| 0.001 | Estable | 93 | 9.7518 |
| 0.01 | Divergente | 1 | 3,959,695,532.5283 |
| 0.1 | Divergente | 1 | 5,012,788,308,558.5518 |
| 1.0 | Divergente | 1 | 53,195,661,488,606.9688 |

La gráfica se guardó en `assets/sgd_learning_rate_exploration.png`.

### Respuestas al enunciado

1. Las tasas que divergieron fueron **0.01, 0.1 y 1.0**. Las tasas `0.0001` y `0.001` se mantuvieron estables durante los 93 chunks.
2. Una tasa grande produce actualizaciones demasiado amplias. El modelo sobrepasa el mínimo de la función de pérdida y los pesos crecen rápidamente, generando divergencia.
3. Una tasa demasiado pequeña es más estable, pero puede hacer que el aprendizaje sea lento y requiera muchas épocas.
4. La tasa debe equilibrar velocidad y estabilidad. En esta corrida, `eta0 = 0.0001` obtuvo el menor RMSE final entre las tasas estables.

### Interpretación

- Una tasa demasiado grande puede causar divergencia porque los pasos exceden el mínimo de la función de pérdida.
- Una tasa demasiado pequeña converge lentamente.
- La tasa adecuada logra una reducción estable del RMSE sin producir `NaN` o valores infinitos.

## 12. Ejercicio 2.5: calendarios de aprendizaje

Se comparan los calendarios `constant`, `optimal` e `invscaling`.

### Resultados obtenidos

| Calendario | Época 1 | Época 2 | Época 3 | RMSE mínimo |
|---|---:|---:|---:|---:|
| `constant` | 9.9486 | 9.9922 | 10.3767 | 9.9486 |
| `optimal` | 684,612,984,085.7560 | 316,378,471,614.8434 | 212,838,042,260.7966 | 212,838,042,260.7966 |
| `invscaling` | 9.6145 | 9.5345 | 9.5421 | 9.5345 |

La gráfica se guardó en `assets/sgd_learning_rate_schedules.png`.

### Respuestas al enunciado

1. El calendario con menor RMSE fue **`invscaling`**, con un valor mínimo de **9.5345 años**.
2. `constant` mantiene una tasa fija y puede adaptarse mejor a cambios en la distribución, pero requiere elegir cuidadosamente `eta0`.
3. `optimal` calcula una tasa decreciente automáticamente. En esta corrida produjo errores extremadamente altos, por lo que la configuración de `alpha` requiere ajuste para este dataset.
4. `invscaling` reduce gradualmente la tasa mediante `power_t=0.25` y obtuvo el mejor resultado.
5. En streaming, una tasa constante puede ser preferible con concept drift; en datos estables, una tasa decreciente puede ofrecer un refinamiento más preciso.

### Interpretación

Las tasas decrecientes pueden mejorar la convergencia en datasets estáticos. En escenarios con cambio de distribución, una tasa constante puede adaptarse mejor a nuevos patrones.

# Resumen final

El laboratorio muestra que la regresión lineal puede representar relaciones complejas cuando se combina con transformaciones, características polinomiales y regularización.

Las conclusiones principales son:

1. Los pipelines reducen el riesgo de fuga de información durante la validación cruzada.
2. Las transformaciones logarítmicas ayudan a controlar la asimetría de las variables.
3. Las características polinomiales mejoran la capacidad del modelo para capturar relaciones no lineales.
4. Ridge es útil cuando existen muchas características correlacionadas y se desea conservarlas.
5. Lasso es útil cuando se busca seleccionar características y construir un modelo más interpretable.
6. El grado polinomial debe seleccionarse mediante validación, porque grados altos pueden causar sobreajuste.
7. `SGDRegressor.partial_fit` permite entrenar modelos con datasets que no caben completamente en memoria.
8. La tasa de aprendizaje controla la estabilidad y velocidad del entrenamiento online.
9. El mejor modelo no se elige únicamente por su error de entrenamiento, sino por su desempeño sobre datos no utilizados durante el ajuste.

## Conclusión del proyecto

El modelo final debe seleccionarse considerando simultáneamente el `Test RMSE`, la estabilidad de la validación cruzada, la complejidad del modelo y la interpretabilidad de sus coeficientes. En la primera comparación realizada, el modelo Ridge de grado 3 obtuvo el mejor resultado entre los grados evaluados, con un `Test RMSE` de 0.5391.
