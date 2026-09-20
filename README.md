# ML Lab 1 — Grupo 8 — Regresión Lineal

Este documento registra, ejercicio por ejercicio, el trabajo realizado sobre
[linear_regression.ipynb](linear_regression.ipynb): de qué parte del notebook se
tomó el código base, qué se añadió o modificó para resolver cada actividad, una
explicación línea por línea del código final, y un resumen de qué hace y para
qué sirve el análisis.

---

## Ejercicio 1.1 — Efecto del Grado Polinómico

### 1. Origen del código

El punto de partida es el pipeline construido en la **Sección 3.3 (Building
the Ridge Pipeline)** del notebook:

```
ColumnTransformer (log1p sobre 5 features + passthrough sobre 3)
        → PolynomialFeatures(degree=POLY_DEGREE)
        → StandardScaler
        → Ridge
```

y el `GridSearchCV` sobre `ridge__alpha` de la **Sección 3.4**, que en el
notebook original se ejecuta **una sola vez** con `POLY_DEGREE = 3` fijo
(definido en la celda de configuración).

### 2. Qué se añadió / modificó

El ejercicio pide comparar los grados `1`, `2` y `3`, algo que el código
original no hace (solo prueba un grado a la vez). Se agregó:

- Un **bucle `for degree in [1, 2, 3]`** que reconstruye el pipeline completo
  en cada iteración (antes se construía una sola vez, fuera de cualquier
  bucle).
- Dentro del bucle, un `GridSearchCV` independiente por grado (misma rejilla
  `ALPHAS` y mismos `CV_FOLDS` que en la Sección 3.4, para que la comparación
  sea justa).
- Cálculo del **Test RMSE** por grado usando `grid_deg.best_estimator_`,
  reutilizando la misma lógica de la Sección 3.5 (`Model Evaluation on the
  Test Set`), que originalmente solo se aplicaba al modelo de grado 3.
- Una lista `degree_results` y un `pd.DataFrame` (`results_df`) para
  **tabular** los resultados — esto no existía antes; se construyó siguiendo
  el mismo estilo que la tabla `results` de la Sección 3.4
  (`pd.DataFrame(grid_search.cv_results_)`).
- Un diccionario `degree_best_estimators` para guardar cada modelo entrenado
  por si se necesita reutilizarlo más adelante (por ejemplo, en el Ejercicio
  1.2 para comparar con Lasso).

### 3. Explicación línea por línea

```python
DEGREES = [1, 2, 3]
degree_results = []
degree_best_estimators = {}
```
Define los tres grados a comparar y dos estructuras vacías: una lista donde
se guardará un diccionario de métricas por grado, y un diccionario para
guardar el mejor modelo (`best_estimator_`) de cada grado.

```python
for degree in DEGREES:
    # Misma receta de preprocesamiento que en la Sección 3.3, solo cambia el grado del polinomio
    log_preprocessor_deg = ColumnTransformer([
        ('log',  FunctionTransformer(np.log1p, feature_names_out='one-to-one'), LOG_FEATURES),
        ('pass', 'passthrough', PASS_FEATURES),
    ], verbose_feature_names_out=False)
```
Por cada grado se crea un `ColumnTransformer` nuevo que aplica `log1p` a las
5 columnas sesgadas (`LOG_FEATURES`) y deja pasar sin cambios las otras 3
(`PASS_FEATURES`). Se recrea en cada vuelta del bucle para evitar reutilizar
un objeto ya ajustado (`fit`) de una iteración anterior.

```python
    pipeline_deg = Pipeline([
        ('log_transform', log_preprocessor_deg),
        ('poly',   PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
        ('scaler', StandardScaler()),
        ('ridge',  Ridge()),
    ])
```
Arma el pipeline de 4 pasos: transformación logarítmica → expansión
polinómica del grado actual → escalado estándar → regresión Ridge. Es
idéntico al de la Sección 3.3, pero con `degree=degree` en vez de la
constante fija `POLY_DEGREE`.

```python
    grid_deg = GridSearchCV(
        pipeline_deg,
        {'ridge__alpha': ALPHAS},
        cv=CV_FOLDS,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    grid_deg.fit(X_train, y_train)
```
Busca, por validación cruzada de `CV_FOLDS` particiones, el mejor `alpha` de
Ridge dentro de la rejilla `ALPHAS` (la misma que en la Sección 3.4). Se usa
`scoring='neg_mean_squared_error'` porque `GridSearchCV` maximiza el score
por convención (de ahí el signo negativo, que luego se revierte con `-`).
`n_jobs=-1` paraleliza el ajuste usando todos los núcleos disponibles.

```python
    best_alpha_deg   = grid_deg.best_params_['ridge__alpha']
    best_cv_rmse_deg = np.sqrt(-grid_deg.best_score_)
```
Extrae el mejor `alpha` encontrado y convierte el mejor score (MSE negativo)
a RMSE real, aplicando `-` y raíz cuadrada.

```python
    y_pred_deg     = grid_deg.best_estimator_.predict(X_test)
    test_rmse_deg  = np.sqrt(mean_squared_error(y_test, y_pred_deg))
    n_features_deg = grid_deg.best_estimator_.named_steps['poly'].n_output_features_
```
Con el mejor modelo (`best_estimator_`, ya reentrenado sobre todo
`X_train`/`y_train` con el mejor `alpha`) se predice sobre el **test set**
(nunca visto durante la búsqueda de hiperparámetros) y se calcula su RMSE.
También se consulta cuántas columnas produjo el paso `poly` de ese pipeline,
para reportar cómo crece la dimensionalidad con el grado.

```python
    degree_best_estimators[degree] = grid_deg.best_estimator_
    degree_results.append({
        'degree':      degree,
        'n_features':  n_features_deg,
        'best_alpha':  best_alpha_deg,
        'cv_rmse':     best_cv_rmse_deg,
        'test_rmse':   test_rmse_deg,
    })
```
Guarda el modelo entrenado y agrega una fila de resultados (grado, número de
features, mejor alpha, RMSE de validación cruzada y RMSE de test) a la lista
que luego se convertirá en tabla.

```python
    print(f'Degree {degree}  ({n_features_deg:>3} features)  '
          f'best α = {best_alpha_deg:8.4f}  |  CV RMSE = {best_cv_rmse_deg:.4f}  |  Test RMSE = {test_rmse_deg:.4f}')
```
Imprime una línea de progreso por cada grado ya evaluado, para poder seguir
el proceso mientras corre (el `GridSearchCV` con 30 valores de alpha y 5
folds tarda varios segundos por grado).

```python
# ── Tabla resumen ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(degree_results).set_index('degree')
print('\nSummary — Ridge CV performance by polynomial degree:')
display(results_df.round(4))
```
Una vez terminado el bucle, convierte la lista de diccionarios en un
`DataFrame` indexado por grado y lo muestra redondeado a 4 decimales — esta
es la "tabla resumen" que pide el enunciado del ejercicio.

```python
best_degree = results_df['cv_rmse'].idxmin()
print(f"\nBest CV RMSE achieved at degree = {best_degree}")
```
Identifica automáticamente qué grado obtuvo el menor error de validación
cruzada, para no tener que leerlo a ojo de la tabla.

### 4. Resultado obtenido

| degree | n_features | best_alpha | cv_rmse | test_rmse |
|---|---|---|---|---|
| 1 | 8   | 2.5929 | 0.6183 | 0.6285 |
| 2 | 44  | 0.0024 | 0.5452 | 0.5513 |
| 3 | 164 | 0.0574 | 0.5336 | 0.5391 |

### 5. Resumen para el análisis

El código automatiza la comparación de tres complejidades de modelo (grados
1, 2 y 3) bajo el **mismo protocolo experimental** (mismo split
train/test, mismos folds de CV, misma rejilla de `alpha`), de modo que las
diferencias en RMSE sean atribuibles únicamente al grado del polinomio y no
a variaciones metodológicas. Por cada grado se ajusta el pipeline completo
dentro de un `GridSearchCV`, se guarda el mejor `alpha` regularizador
encontrado y se mide el error tanto en validación cruzada (`cv_rmse`, usado
para elegir el `alpha`) como en el conjunto de test totalmente independiente
(`test_rmse`, usado para juzgar la generalización real).

Los resultados muestran una mejora **monótona pero con retornos
decrecientes**: pasar de grado 1 a 2 reduce el Test RMSE en 0.077 (una mejora
grande, porque el modelo lineal puro tiene alto sesgo), mientras que pasar de
grado 2 a 3 solo lo reduce en 0.012, a pesar de que el número de features
casi se cuadruplica (44 → 164). Esto evidencia el compromiso
**sesgo–varianza**: `GridSearchCV` compensa la varianza extra de cada grado
adicional escogiendo un `alpha` de regularización mayor, pero esa
compensación tiene un límite — más allá de cierto grado, agregar más
features deja de traducirse en mejor generalización y el análisis concluye
que **no** es correcto asumir que "mayor grado siempre implica mejor
desempeño".

---

## Ejercicio 1.2 — Ridge vs. Lasso

### 1. Origen del código

Se reutiliza otra vez la misma receta de preprocesamiento de la **Sección
3.3** (`ColumnTransformer` con `log1p` sobre `LOG_FEATURES` + passthrough
sobre `PASS_FEATURES`) y el mismo esquema de búsqueda de hiperparámetros de
la **Sección 3.4** (`GridSearchCV` sobre la rejilla `ALPHAS`). El enunciado
pide explícitamente comparar "los coeficientes de un polinomio de grado 2",
así que a diferencia del Ejercicio 1.1 (que barría varios grados), aquí el
grado queda fijo en `POLY_DEGREE_EX12 = 2` para ambos modelos.

### 2. Qué se añadió / modificó

- Se construyó un **segundo pipeline en paralelo**: uno con `Ridge` (idéntico
  en estructura al de la Sección 3.3, solo que en grado 2) y otro con
  `Lasso` en el mismo lugar del último paso — exactamente lo que pide la
  tarea 1 ("misma estructura, cambia el estimador final").
- Se añadió `max_iter=20_000` a `Lasso` porque, a diferencia de `Ridge` (que
  tiene solución cerrada), Lasso se resuelve por descenso de coordenadas y
  con `alpha` muy pequeño puede no converger con el `max_iter` por defecto
  (1000).
- Se agregó el **conteo de coeficientes exactamente en cero**
  (`np.sum(coefs == 0)`) para ambos modelos — esta comparación no existe en
  ninguna parte anterior del notebook, es nueva para esta actividad.
- Se agregó la extracción de **nombres de features reales** a partir de
  `poly.get_feature_names_out(LOG_FEATURES + PASS_FEATURES)`. Por defecto,
  como el `Pipeline` pasa arrays de NumPy entre pasos (no DataFrames), este
  método devuelve nombres genéricos (`x0`, `x1`, ...); pasarle explícitamente
  el orden de columnas que produce el `ColumnTransformer` corrige eso y deja
  ver, por ejemplo, `Latitude Longitude` en vez de `x6 x7`.
- Se armó el **gráfico de barras horizontales** de los 20 coeficientes no
  nulos más grandes en valor absoluto (tarea 4), coloreando en rojo los
  negativos y en azul los positivos, siguiendo la misma paleta `COLORS` que
  el resto del notebook.

### 3. Explicación línea por línea

```python
POLY_DEGREE_EX12 = 2
```
Fija el grado en 2 porque la tarea 3 pide contar coeficientes "del
polinomio de grado 2" — no se reutiliza `POLY_DEGREE` (que vale 3 en la
configuración global) para no mezclar resultados de secciones distintas.

```python
log_preprocessor_ex12 = ColumnTransformer([
    ('log',  FunctionTransformer(np.log1p, feature_names_out='one-to-one'), LOG_FEATURES),
    ('pass', 'passthrough', PASS_FEATURES),
], verbose_feature_names_out=False)
```
Mismo transformador de la Sección 3.3: aplica `log1p` a las 5 columnas
sesgadas y deja pasar las otras 3 sin cambios. Se crea una sola vez y se
reutiliza en ambos pipelines (Ridge y Lasso) — es seguro porque
`GridSearchCV`/`Pipeline` clona internamente cada paso antes de ajustarlo,
así que no hay interferencia entre los dos modelos.

```python
ridge_pipeline_ex12 = Pipeline([
    ('log_transform', log_preprocessor_ex12),
    ('poly',   PolynomialFeatures(degree=POLY_DEGREE_EX12, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge',  Ridge()),
])

ridge_grid_ex12 = GridSearchCV(
    ridge_pipeline_ex12,
    {'ridge__alpha': ALPHAS},
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)
ridge_grid_ex12.fit(X_train, y_train)
```
Arma el pipeline de referencia (Ridge, grado 2) y busca el mejor `alpha` por
validación cruzada, exactamente igual que en la Sección 3.4 pero con la
rejilla ya conocida `ALPHAS`.

```python
lasso_pipeline_ex12 = Pipeline([
    ('log_transform', log_preprocessor_ex12),
    ('poly',   PolynomialFeatures(degree=POLY_DEGREE_EX12, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lasso',  Lasso(max_iter=20_000)),
])

lasso_grid_ex12 = GridSearchCV(
    lasso_pipeline_ex12,
    {'lasso__alpha': ALPHAS},
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)
lasso_grid_ex12.fit(X_train, y_train)
```
Mismo pipeline, cambiando únicamente el último paso por `Lasso`. `max_iter`
se sube a 20 000 para darle margen al optimizador de converger en los
valores de `alpha` más pequeños de la rejilla.

```python
ridge_best_ex12 = ridge_grid_ex12.best_estimator_
lasso_best_ex12 = lasso_grid_ex12.best_estimator_
```
Guarda el mejor pipeline entrenado de cada modelo (ya reajustado sobre todo
`X_train` con su mejor `alpha`), para usarlos en las tareas siguientes.

```python
ridge_test_rmse_ex12 = np.sqrt(mean_squared_error(y_test, ridge_best_ex12.predict(X_test)))
lasso_test_rmse_ex12 = np.sqrt(mean_squared_error(y_test, lasso_best_ex12.predict(X_test)))
```
**Tarea 2**: calcula el RMSE de cada modelo sobre el conjunto de test
(nunca usado para elegir `alpha`), para comparar su desempeño real.

```python
ridge_coefs_ex12 = ridge_best_ex12.named_steps['ridge'].coef_
lasso_coefs_ex12 = lasso_best_ex12.named_steps['lasso'].coef_
n_total_coefs_ex12 = len(lasso_coefs_ex12)

n_ridge_zero_ex12 = int(np.sum(ridge_coefs_ex12 == 0))
n_lasso_zero_ex12 = int(np.sum(lasso_coefs_ex12 == 0))
```
**Tarea 3**: extrae el arreglo de coeficientes (`coef_`) de cada modelo
entrenado y cuenta cuántos son exactamente `0`. Como ambos pipelines
comparten el mismo `PolynomialFeatures(degree=2)`, tienen el mismo número
total de coeficientes (`n_total_coefs_ex12`), lo que hace la comparación
directa.

```python
input_feature_names_ex12 = LOG_FEATURES + PASS_FEATURES
feature_names_ex12 = lasso_best_ex12.named_steps['poly'].get_feature_names_out(input_feature_names_ex12)
```
El `ColumnTransformer` (con `verbose_feature_names_out=False`) produce sus
columnas en este orden: primero las de `LOG_FEATURES` (transformadas),
después las de `PASS_FEATURES`. Pasar esa misma lista como
`input_features` a `get_feature_names_out` es lo que permite que
`PolynomialFeatures` devuelva nombres legibles (`MedInc`, `Latitude
Longitude`, `MedInc^2`, ...) en vez de los genéricos `x0`, `x1`, ... que
usaría por defecto al no recibir un DataFrame con nombres de columnas.

```python
lasso_coef_df_ex12 = pd.DataFrame({
    'feature': feature_names_ex12,
    'coef':    lasso_coefs_ex12,
})
lasso_nonzero_ex12 = lasso_coef_df_ex12[lasso_coef_df_ex12['coef'] != 0].copy()
lasso_nonzero_ex12['abs_coef'] = lasso_nonzero_ex12['coef'].abs()
top20_ex12 = lasso_nonzero_ex12.sort_values('abs_coef', ascending=False).head(20).sort_values('coef')
```
**Tarea 4** (parte 1): arma una tabla feature ↔ coeficiente, descarta los
que son cero, calcula el valor absoluto para ordenar por magnitud, se queda
con los 20 más grandes y los reordena por valor (no por magnitud) para que
el gráfico de barras horizontales quede ordenado de negativo a positivo.

```python
fig, ax = plt.subplots(figsize=(9, 7))
bar_colors_ex12 = [COLORS[1] if c < 0 else COLORS[0] for c in top20_ex12['coef']]
ax.barh(top20_ex12['feature'], top20_ex12['coef'], color=bar_colors_ex12)
ax.axvline(0, color='black', lw=0.8)
...
plt.savefig('assets/lasso_top20_coefficients.png', dpi=120, bbox_inches='tight')
plt.show()
```
**Tarea 4** (parte 2): dibuja el gráfico de barras horizontales pedido,
coloreando en rojo los coeficientes negativos y en azul los positivos, con
una línea vertical en 0 como referencia — mismo estilo visual (`COLORS`,
`plt.savefig` en `assets/`) que el resto del notebook.

### 4. Resultado obtenido

| Modelo | Grado | Best α | Test RMSE | Coeficientes en cero |
|---|---|---|---|---|
| Ridge | 2 | 0.0024 | 0.5513 | 0 / 44 |
| Lasso | 2 | 0.0001 | 0.5634 | 9 / 44 (20.5%) |

El coeficiente Lasso de mayor magnitud es `Latitude × Longitude`, seguido de
`Latitude²`, `MedInc × Population` y `MedInc²`.

### 5. Resumen para el análisis

Ridge y Lasso se entrenaron sobre el **mismo pipeline y el mismo conjunto de
44 features de grado 2**, cambiando únicamente el tipo de penalización (L2
vs. L1). Ridge terminó con un Test RMSE ligeramente menor (0.5513 vs.
0.5634) y **no anuló ningún coeficiente**: su penalización cuadrática reparte
el peso entre todas las features, incluidas las poco informativas o
correlacionadas entre sí, aprovechando cualquier señal por pequeña que sea.
Lasso, en cambio, **apagó por completo 9 de los 44 coeficientes (20.5%)**,
porque su penalización L1 tiene un punto en el que, para una feature poco
útil, es matemáticamente más barato llevarla a cero que dejarla con un valor
pequeño.

Esto convierte a Lasso en un mecanismo de **selección automática de
features**: en vez de usar las 44 columnas generadas por
`PolynomialFeatures(degree=2)`, el modelo se queda solo con 35 y descarta el
resto sin intervención manual. El gráfico de coeficientes muestra que las
que sobrevivieron con más peso tienen sentido de negocio: la interacción
`Latitude × Longitude` (la ubicación geográfica combinada) es, por lejos, el
término más importante, seguida de interacciones con `MedInc` — confirmando
que el precio de una vivienda no depende del ingreso o de la ubicación por
separado, sino de cómo ambos interactúan. La contrapartida es que, ante
features correlacionadas, Lasso tiende a quedarse con una y descartar la
otra de forma casi arbitraria, mientras que Ridge las reparte de forma más
estable — lo que probablemente explica por qué Ridge generalizó levemente
mejor en este caso. En síntesis: Lasso conviene cuando se prioriza
interpretabilidad y selección de variables; Ridge, cuando se prioriza
estabilidad y el mejor desempeño posible.

---

## Parte 2 — Online Learning con SGDRegressor

### Ejercicio 2.1 — Implementar el bucle de entrenamiento mini-batch

#### 1. Origen del código

Se parte de dos piezas ya presentes en el notebook:

- El ejemplo de código de la **Sección 4.3** (`partial_fit` en sklearn), que
  muestra el patrón general: iterar con `pd.read_csv(..., chunksize=...)`,
  separar `X_chunk`/`y_chunk` y llamar a `sgd.partial_fit(X_scaled,
  y_chunk)`.
- Las constantes ya definidas en la celda de configuración inicial:
  `CHUNK_SIZE = 5_000`, `MSD_TRAIN_ROWS = 463_715`, `MSD_DATA_FILE`,
  `RANDOM_STATE`.

Ninguna celda anterior implementaba el bucle completo — la Sección 4.3 solo
lo mostraba como fragmento ilustrativo (`for chunk in
pd.read_csv('bigfile.csv', ...)`), sin ejecutarlo ni separar las dos pasadas
(scaler vs. entrenamiento).

#### 2. Qué se añadió / modificó

- Se creó el estimador `sgd = SGDRegressor(loss='squared_error',
  learning_rate='constant', eta0=0.001, random_state=42)` y el
  `msd_scaler = StandardScaler()`, tal como pide la tarea 1 y 2.
- Se implementó la **Pasada 1** completa: un bucle que recorre *todos* los
  chunks de entrenamiento llamando únicamente a
  `msd_scaler.partial_fit(X_chunk)` — sin tocar todavía el modelo — para que
  el scaler vea la distribución completa antes de escalar ningún dato.
- Se implementó la **Pasada 2**: un segundo recorrido (con un iterador
  `pd.read_csv` nuevo, ya que un iterador de chunks no se puede reutilizar
  una vez consumido) donde cada chunk se transforma con el scaler ya
  ajustado y se entrena el `SGDRegressor` con `partial_fit`.
- Se añadió el cálculo de **RMSE por chunk** justo después de cada
  actualización de pesos, guardado en la lista `chunk_rmses` — esta lista no
  existía antes y es la que se reutilizará en el Ejercicio 2.2 para graficar
  la curva de convergencia.

#### 3. Explicación línea por línea

```python
sgd = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=0.001, random_state=RANDOM_STATE)
msd_scaler = StandardScaler()
```
Crea el regresor SGD con pérdida cuadrática (equivalente a OLS) y tasa de
aprendizaje **constante** (no decae con el tiempo), y un `StandardScaler`
independiente que se ajustará de forma incremental (`partial_fit`) porque
los datos nunca se cargan completos en memoria.

```python
train_reader_pass1 = pd.read_csv(MSD_DATA_FILE, header=None, chunksize=CHUNK_SIZE, nrows=MSD_TRAIN_ROWS)
for chunk in train_reader_pass1:
    X_chunk = chunk.iloc[:, 1:].values
    msd_scaler.partial_fit(X_chunk)
```
`pd.read_csv(..., chunksize=CHUNK_SIZE)` devuelve un iterador perezoso: cada
vuelta del `for` lee solo `CHUNK_SIZE` filas del disco, nunca el archivo
completo. `nrows=MSD_TRAIN_ROWS` corta la lectura exactamente en la fila
463,715 para no mezclar datos de test. `chunk.iloc[:, 1:]` descarta la
columna 0 (el año, la variable objetivo) y deja las 90 columnas de audio.
`partial_fit` actualiza incrementalmente la media y varianza del scaler sin
necesitar ver los datos anteriores otra vez.

```python
print(f'Scaler ajustado sobre {MSD_TRAIN_ROWS:,} filas de entrenamiento (todas las columnas 1–90).')
```
Confirma que la primera pasada terminó antes de empezar a entrenar el
modelo — importante porque si se usara el scaler a medio ajustar, cada
chunk se escalaría con estadísticas distintas (parciales), introduciendo
ruido en el entrenamiento.

```python
chunk_rmses = []
train_reader_pass2 = pd.read_csv(MSD_DATA_FILE, header=None, chunksize=CHUNK_SIZE, nrows=MSD_TRAIN_ROWS)
for chunk in train_reader_pass2:
    y_chunk = chunk.iloc[:, 0].values
    X_chunk = chunk.iloc[:, 1:].values
    X_scaled = msd_scaler.transform(X_chunk)

    sgd.partial_fit(X_scaled, y_chunk)
```
Se crea un **segundo** iterador de `read_csv` (el primero ya se consumió por
completo en la Pasada 1 y no se puede "rebobinar"). Por cada chunk: separa
la variable objetivo (`year`, columna 0) de las features, escala con
`transform` (no `fit_transform`, porque el scaler ya quedó fijo en la
Pasada 1) y actualiza los pesos del modelo con ese lote mediante
`partial_fit`, sin olvidar lo aprendido en chunks anteriores.

```python
    y_pred_chunk = sgd.predict(X_scaled)
    chunk_rmse = np.sqrt(mean_squared_error(y_chunk, y_pred_chunk))
    chunk_rmses.append(chunk_rmse)
```
Justo después de actualizar los pesos con este chunk, se predice sobre el
**mismo** chunk y se mide el RMSE. No es una medida de generalización (es el
mismo lote que se acaba de usar para entrenar), pero sirve como indicador de
qué tan bien está ajustando el modelo a medida que avanza — es la señal que
se grafica en el Ejercicio 2.2.

```python
print(f'\nChunks procesados : {len(chunk_rmses)}')
print(f'RMSE del último chunk: {chunk_rmses[-1]:.4f} años')
```
**Tarea 5**: reporta cuántos chunks se procesaron en total y el error del
último de ellos, como resumen final de la pasada de entrenamiento.

#### 4. Resultado obtenido

```
Scaler ajustado sobre 463,715 filas de entrenamiento (todas las columnas 1–90).

Chunks procesados : 93
RMSE del último chunk: 9.7518 años
```

El enunciado sugiere "~10–11 chunks" como resultado esperado, pero con el
`CHUNK_SIZE = 5_000` ya definido en la celda de configuración del notebook,
recorrer las 463,715 filas de entrenamiento produce **93 chunks**
(`463715 / 5000 ≈ 92.7`, redondeado hacia arriba). El RMSE final (9.75 años)
sí cae dentro del rango esperado (8–11 años), así que el entrenamiento se
comporta correctamente — el número de chunks distinto es solo una
consecuencia del tamaño de lote configurado, no un error.

#### 5. Resumen para el análisis

Este ejercicio implementa la versión más básica de **aprendizaje online**:
en vez de cargar los ~464 mil registros y 90 features en memoria de una
sola vez, el modelo se entrena leyendo el archivo en fragmentos de 5,000
filas, actualizando sus pesos incrementalmente con `partial_fit` y
descartando cada fragmento apenas se usa. La clave metodológica es separar
el ajuste del `StandardScaler` (que necesita ver la distribución completa
para calcular una media/varianza representativa) del entrenamiento del
modelo (que si empezara a recibir datos mal escalados desde el primer chunk,
partiría con gradientes distorsionados). Haciendo una pasada dedicada solo
para el scaler antes de tocar el modelo, se evita ese problema sin necesitar
cargar el dataset completo en RAM en ningún momento.

El resultado — un RMSE que baja hasta ~9.75 años en el último chunk de una
sola época — muestra que incluso con una sola pasada por los datos, mini-batch
SGD ya logra un ajuste razonable (recordando que el rango de años del target
es 1922–2011, un error de ~10 años es una fracción moderada de ese rango).
El comportamiento chunk a chunk de este error es precisamente lo que se
visualiza en el Ejercicio 2.2 para ver si el modelo converge dentro de la
misma época.

### Ejercicio 2.2 — Graficar la curva de convergencia

#### 1. Origen del código

Se reutiliza directamente la lista `chunk_rmses` construida en el Ejercicio
2.1 — este ejercicio no entrena nada nuevo, solo visualiza un resultado que
ya existía. El estilo del gráfico (figura, colores de `COLORS`, `savefig` en
`assets/`) sigue el mismo patrón que el resto de gráficos del notebook (por
ejemplo, la curva de validación de la Sección 3.4).

#### 2. Qué se añadió / modificó

- Un gráfico de línea (`RMSE del chunk` vs. `número de chunk`) que no existía
  antes, usando `np.arange(1, len(chunk_rmses) + 1)` como eje X para que el
  primer chunk sea el 1 y no el 0 (más natural de leer).
- Impresión de tres estadísticos de apoyo (RMSE del primer chunk, del último,
  y el mínimo con su posición) para poder comentar el patrón con números
  concretos en vez de solo "a ojo" sobre el gráfico.

#### 3. Explicación línea por línea

```python
chunk_idx = np.arange(1, len(chunk_rmses) + 1)
ax.plot(chunk_idx, chunk_rmses, color=COLORS[0], marker='o', markersize=3, lw=1.5)
```
Genera el eje X (1, 2, 3, ..., 93) y grafica la lista de RMSEs guardada
durante el entrenamiento de la Sección 2.1. Se usan marcadores pequeños
además de la línea para poder distinguir picos aislados de una tendencia
suave.

```python
print(f'RMSE mínimo       : {min(chunk_rmses):.4f} años (chunk {int(np.argmin(chunk_rmses)) + 1})')
```
`np.argmin` devuelve la posición (índice base 0) del menor RMSE de la lista;
se le suma 1 para reportarlo en la misma numeración de chunks que el eje X
del gráfico (base 1).

#### 4. Resultado obtenido

- RMSE primer chunk: **44.24** años (pesos inicializados en cero).
- RMSE último chunk: **9.75** años.
- RMSE mínimo: **7.72** años, en el chunk 53.
- Dos picos aislados: uno cerca del chunk 19 (≈34.6) y otro cerca del chunk
  92 (≈58.0).

#### 5. Resumen para el análisis

La curva confirma que el modelo **sí converge dentro de una sola época**, y
lo hace casi de inmediato: en 4-5 chunks el RMSE ya cayó de 44 a la banda
estable de 8-11 años, y se mantiene ahí durante el resto del entrenamiento
sin una tendencia clara a seguir bajando. Esto tiene sentido porque el
problema es esencialmente lineal (predecir un año a partir de estadísticas
de timbre) y `eta0 = 0.001` es lo bastante grande como para que unos pocos
miles de muestras alcancen para acercarse al óptimo.

Lo más interesante son los dos picos puntuales que interrumpen esa
estabilidad. La explicación más probable es que el archivo del dataset
**no está barajado**: como se lee secuencialmente con `pd.read_csv`, cada
chunk es un bloque contiguo de 5,000 filas tal como viene ordenado en el
archivo original, y ese orden no es aleatorio respecto a la distribución de
años/timbres. Si un chunk cae, por la estructura del archivo, en un tramo
poco representativo (por ejemplo, muchas canciones de un estilo o época
inusual), el modelo entrenado hasta ese punto falla fuerte en ese lote
puntual — pero se recupera de inmediato porque los chunks siguientes vuelven
a ser representativos de la distribución general. Este hallazgo motiva
directamente el **Ejercicio 2.3**, donde se agrega `shuffle` de los datos al
inicio de cada época precisamente para evitar este tipo de picos causados
por el orden del archivo.
### Ejercicio 2.3 — Entrenamiento multi-época

#### 1. Origen del código

Reutiliza directamente el `msd_scaler` ya ajustado en el Ejercicio 2.1 (no
se vuelve a llamar a `partial_fit` sobre él, solo a `transform`) y el mismo
patrón de `SGDRegressor` + `partial_fit` por chunk. La diferencia central
frente al Ejercicio 2.1 es que aquí el bucle se repite `N_EPOCHS` veces (ya
definido en la configuración inicial, `N_EPOCHS = 10`) y cada época procesa
los datos en un orden distinto.

#### 2. Qué se añadió / modificó

- Se agregó la **carga del conjunto de test** (`pd.read_csv(...,
  skiprows=MSD_TRAIN_ROWS)`), algo que no existía en el Ejercicio 2.1 porque
  ahí solo se entrenaba, sin medir generalización.
- Para poder **barajar** el orden de las filas en cada época, se cargó el
  train set completo en memoria una sola vez (`train_df_ex23`) en vez de
  volver a leer el CSV por chunks con `pd.read_csv(chunksize=...)` como en
  el Ejercicio 2.1. Esto es necesario porque un iterador de `read_csv` no se
  puede "rebarajar" sin releer el archivo completo — es más simple generar
  una permutación de índices con NumPy y cortarla en bloques de
  `CHUNK_SIZE`.
- Se creó un **`SGDRegressor` nuevo** (`sgd_multi`), distinto al `sgd` del
  Ejercicio 2.1, para no mezclar el modelo de una época con el de
  entrenamiento multi-época.
- Se añadieron dos listas de seguimiento: `train_chunk_rmses_multi` (RMSE de
  cada chunk, con las 10 épocas concatenadas una detrás de otra) y
  `val_rmses_per_epoch` (un solo RMSE de validación por época, calculado
  sobre el test set fijo).
- Se agregó el gráfico combinado (tarea 3): una curva de train por chunk
  (con opacidad reducida por el volumen de puntos) y, superpuestos en los
  mismos ejes, los puntos de validación ubicados exactamente en el chunk
  donde termina cada época — algo que no aparecía en ningún gráfico anterior
  del notebook.

#### 3. Explicación línea por línea

```python
test_df_ex23 = pd.read_csv(MSD_DATA_FILE, header=None, skiprows=MSD_TRAIN_ROWS)
y_test_msd = test_df_ex23.iloc[:, 0].values
X_test_msd = test_df_ex23.iloc[:, 1:].values
X_test_msd_scaled = msd_scaler.transform(X_test_msd)
```
`skiprows=MSD_TRAIN_ROWS` salta las primeras 463,715 filas (todo el train) y
lee el resto del archivo, que corresponde exactamente a las 51,630 filas de
test según el split oficial del dataset. Se transforma con el `msd_scaler`
ya congelado — nunca se llama `fit`/`partial_fit` sobre datos de test, para
no filtrar información del conjunto de evaluación al entrenamiento.

```python
train_df_ex23 = pd.read_csv(MSD_DATA_FILE, header=None, nrows=MSD_TRAIN_ROWS)
y_train_msd = train_df_ex23.iloc[:, 0].values
X_train_msd = train_df_ex23.iloc[:, 1:].values
n_train_msd = X_train_msd.shape[0]
```
Carga todo el conjunto de entrenamiento en memoria de una sola vez (~464 mil
filas × 90 columnas caben sin problema en RAM). Esto es una simplificación
razonable frente al streaming estricto del Ejercicio 2.1: aquí el objetivo
es poder generar un orden distinto por época, y hacerlo sobre un array ya
cargado es mucho más simple que barajar un archivo en disco.

```python
sgd_multi = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=0.001, random_state=RANDOM_STATE)
rng_ex23 = np.random.default_rng(RANDOM_STATE)
```
Mismos hiperparámetros que el `sgd` del Ejercicio 2.1, para que la
comparación entre "1 época" y "10 épocas" sea justa. `rng_ex23` es un
generador de números aleatorios propio (con semilla fija) que se usará para
barajar los índices en cada época de forma reproducible.

```python
for epoch in range(N_EPOCHS):
    shuffled_idx = rng_ex23.permutation(n_train_msd)

    for start in range(0, n_train_msd, CHUNK_SIZE):
        batch_idx = shuffled_idx[start:start + CHUNK_SIZE]
        X_chunk = msd_scaler.transform(X_train_msd[batch_idx])
        y_chunk = y_train_msd[batch_idx]

        sgd_multi.partial_fit(X_chunk, y_chunk)
```
`rng_ex23.permutation(n_train_msd)` genera un reordenamiento aleatorio
distinto **en cada vuelta del `for epoch`** — esta es la línea que resuelve
la tarea de "barajar los datos al inicio de cada época". El bucle interno
recorta esa permutación en bloques de `CHUNK_SIZE` índices y arma cada
mini-batch indexando directamente sobre el array ya cargado
(`X_train_msd[batch_idx]`), en vez de leer del disco como en el Ejercicio
2.1.

```python
        chunk_rmse = np.sqrt(mean_squared_error(y_chunk, sgd_multi.predict(X_chunk)))
        train_chunk_rmses_multi.append(chunk_rmse)
```
Igual que en el Ejercicio 2.1: mide el RMSE del propio chunk justo después
de la actualización, y lo va acumulando en una sola lista continua a lo
largo de las 10 épocas (por eso el eje X del gráfico final se llama "chunk,
todas las épocas concatenadas").

```python
    y_val_pred = sgd_multi.predict(X_test_msd_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_test_msd, y_val_pred))
    val_rmses_per_epoch.append(val_rmse)
```
Al terminar cada época completa (después de procesar los ~93 chunks), se
evalúa el modelo sobre el conjunto de test — este sí es un RMSE de
**generalización real**, a diferencia del RMSE por chunk que se mide sobre
datos ya usados para entrenar.

```python
n_chunks_per_epoch = int(np.ceil(n_train_msd / CHUNK_SIZE))
epoch_end_positions = [(e + 1) * n_chunks_per_epoch for e in range(N_EPOCHS)]
```
Calcula en qué posición del eje X (número de chunk acumulado) termina cada
época, para poder ubicar los puntos de `val_rmses_per_epoch` exactamente
alineados con el final de la época correspondiente en el mismo gráfico que
la curva de train.

```python
ax.plot(chunk_idx_multi, train_chunk_rmses_multi, color=COLORS[0], lw=1, alpha=0.6, ...)
ax.plot(epoch_end_positions, val_rmses_per_epoch, color=COLORS[1], marker='o', lw=2, ...)
```
**Tarea 3**: dibuja ambas curvas en los mismos ejes — el train con opacidad
reducida (hay ~930 puntos, una línea sólida saturaría el gráfico) y la
validación con marcadores grandes y bien visibles, ya que son solo 10
puntos.

#### 4. Resultado obtenido

```
Test set cargado : 51,630 filas
Train set cargado: 463,715 filas
Época  1/10  →  Val RMSE = 9.9274 años
Época  2/10  →  Val RMSE = 10.0818 años
Época  3/10  →  Val RMSE = 10.0053 años
Época  4/10  →  Val RMSE = 9.8727 años
Época  5/10  →  Val RMSE = 9.8559 años
Época  6/10  →  Val RMSE = 9.9298 años
Época  7/10  →  Val RMSE = 10.3874 años
Época  8/10  →  Val RMSE = 9.7426 años
Época  9/10  →  Val RMSE = 9.7712 años
Época 10/10 →  Val RMSE = 9.8436 años
```

#### 5. Resumen para el análisis

El resultado más importante es que el RMSE de validación **no mejora de
forma consistente** entre épocas: se mueve dentro de una banda estrecha de
~9.7 a ~10.4 años desde la primera época, con la época 7 como un pico
aislado y la época 8 como el mejor valor — una diferencia tan chica que bien
podría ser ruido de SGD y no una mejora real del ajuste. En otras palabras,
el modelo **ya satura en la época 1**: la primera pasada completa alcanza
casi de inmediato la región de error que esta configuración es capaz de
lograr, y las siguientes nueve épocas no aportan información nueva — solo
hacen que el modelo dé vueltas alrededor del mismo nivel de error.

Esto es coherente con lo observado en el Ejercicio 2.2 (el error de train ya
caía a la banda estable de ~9-10 años en los primeros chunks de una sola
época) y expone la limitación de usar una tasa de aprendizaje **constante**:
como `eta0` nunca decae, el modelo nunca deja de dar "pasos grandes" en cada
actualización, así que en vez de ir afinando el ajuste progresivamente,
oscila alrededor del mínimo sin terminar de asentarse en él. Ese es
justamente el problema que exploran los Ejercicios 2.4 (impacto de `eta0`)
y 2.5 (*schedules* que sí decaen con el tiempo).

---

### Ejercicio 2.4 — Exploración del Learning Rate

#### 1. Origen del código

Reutiliza `X_train_msd`, `y_train_msd` (ya cargados en memoria en el
Ejercicio 2.3) y `msd_scaler` (ya ajustado en el Ejercicio 2.1), evitando
releer el CSV una quinta vez. La estructura del bucle interno (recorrer
`X_train_msd`/`y_train_msd` en bloques de `CHUNK_SIZE` y llamar
`partial_fit`) es la misma que en el Ejercicio 2.1, pero sin la fase de
ajuste del scaler (ya está hecha) y sin barajar (para que las 5 corridas
sean directamente comparables entre sí, viendo los datos en el mismo
orden).

#### 2. Qué se añadió / modificó

- Un **bucle externo sobre `ETA0_VALUES = [0.0001, 0.001, 0.01, 0.1,
  1.0]`**, que entrena un `SGDRegressor` distinto por cada valor — esto no
  existía antes, es el corazón de este ejercicio.
- Un **criterio de divergencia** más robusto que el literal "RMSE es NaN o
  inf" del enunciado: en la práctica, con `eta0` grande el RMSE no llega a
  `inf` en la primera actualización, sino que salta a valores absurdamente
  grandes pero técnicamente finitos (`~4e9`, `~5e12`, `~5e13`). Se agregó
  `DIVERGENCE_THRESHOLD = 10_000` años como corte adicional: muy por encima
  del "ruido" normal de warm-up del primer chunk (que puede llegar a
  ~1,000-2,000 años porque los pesos parten en cero), pero muy por debajo de
  una divergencia real.
- Un **`break` temprano** dentro del bucle de chunks apenas se detecta
  divergencia, para no seguir entrenando (ni gastar tiempo) un modelo que ya
  explotó.
- El **diccionario `lr_results`**, que guarda por cada `eta0` la lista de
  RMSEs, si divergió y en qué chunk — estructura nueva, pensada para poder
  graficar después las 5 corridas sin tener que re-entrenar nada.
- El gráfico con las 5 curvas superpuestas y el eje Y recortado a 150 años
  (tarea 3), usando directamente los 5 colores de la paleta `COLORS` ya
  definida (por coincidencia son exactamente 5).

#### 3. Explicación línea por línea

```python
ETA0_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0]
DIVERGENCE_THRESHOLD = 10_000
```
Los 5 valores de `eta0` que pide el enunciado, y el umbral (en años de RMSE)
que se usa para decidir si una corrida "explotó" en la práctica, aunque
matemáticamente el número siga siendo finito.

```python
for eta0 in ETA0_VALUES:
    sgd_lr = SGDRegressor(loss='squared_error', learning_rate='constant', eta0=eta0, random_state=RANDOM_STATE)
    rmses = []
    diverged = False
    diverged_at = None
```
Por cada learning rate se crea un modelo **nuevo desde cero** (pesos en
cero), para que la comparación entre corridas sea justa — ninguna arranca
con ventaja sobre otra.

```python
    for start in range(0, n_train_msd, CHUNK_SIZE):
        X_chunk = msd_scaler.transform(X_train_msd[start:start + CHUNK_SIZE])
        y_chunk = y_train_msd[start:start + CHUNK_SIZE]

        sgd_lr.partial_fit(X_chunk, y_chunk)

        chunk_rmse = np.sqrt(mean_squared_error(y_chunk, sgd_lr.predict(X_chunk)))
        rmses.append(chunk_rmse)
```
Recorre el array ya cargado en memoria en bloques de `CHUNK_SIZE` (sin
barajar, a diferencia del Ejercicio 2.3), escala cada bloque con el
`msd_scaler` congelado, actualiza los pesos con `partial_fit` y mide el RMSE
del propio chunk justo después — igual que en el Ejercicio 2.1, pero
repetido 5 veces con distinto `eta0`.

```python
        if not np.isfinite(chunk_rmse) or chunk_rmse > DIVERGENCE_THRESHOLD:
            diverged = True
            diverged_at = len(rmses)
            break
```
**Tarea 2**: si el RMSE deja de ser finito (`NaN`/`inf`) **o** supera el
umbral de divergencia práctica, se marca el modelo como divergido, se anota
en qué chunk pasó y se corta el entrenamiento de ese modelo de inmediato —
no tiene sentido seguir entrenando algo que ya explotó.

```python
lr_results[eta0] = {'rmses': rmses, 'diverged': diverged, 'diverged_at': diverged_at}
```
Guarda todo lo necesario para graficar y reportar después, indexado por el
valor de `eta0` usado en esa corrida.

```python
for eta0, color in zip(ETA0_VALUES, COLORS):
    rmses = lr_results[eta0]['rmses']
    x_vals = np.arange(1, len(rmses) + 1)
    label = f'eta0 = {eta0}'
    if lr_results[eta0]['diverged']:
        label += '  (divergió)'
    ax.plot(x_vals, rmses, color=color, lw=1.8, marker='o', markersize=3, label=label)
```
**Tarea 3**: dibuja las 5 curvas en los mismos ejes, una por cada `eta0`,
usando el color correspondiente de la paleta del notebook y marcando en la
leyenda cuáles divergieron. Las corridas divergentes solo tienen 1-2 puntos
(se cortó el entrenamiento apenas se detectó la explosión), así que
aparecen como un punto aislado fuera del rango visible del gráfico.

```python
ax.set_ylim(0, 150)
```
Recorta el eje Y a 150 años, tal como pide el enunciado, para que las
corridas divergentes (con RMSEs de miles de millones) no aplasten visualmente
la escala y no se puedan comparar las dos curvas que sí convergen.

#### 4. Resultado obtenido

```
eta0 = 0.0001   →  completó la época sin diverger  (RMSE final = 9.4396 años)
eta0 = 0.001    →  completó la época sin diverger  (RMSE final = 9.7518 años)
eta0 = 0.01     →  DIVERGIÓ en el chunk 1 / 93  (RMSE = 3.96e+09 años)
eta0 = 0.1      →  DIVERGIÓ en el chunk 1 / 93  (RMSE = 5.01e+12 años)
eta0 = 1.0      →  DIVERGIÓ en el chunk 1 / 93  (RMSE = 5.32e+13 años)
```

#### 5. Resumen para el análisis

Solo los dos learning rates más pequeños de los cinco probados (`0.0001` y
`0.001`) logran entrenar sin explotar; los otros tres (`0.01`, `0.1`, `1.0`)
divergen en la primerísima actualización de pesos, con RMSEs que van de
miles de millones a decenas de billones de años — números sin ningún
sentido físico para un problema cuyo target vive entre 1922 y 2011.

La causa es geométrica: `SGDRegressor` actualiza los pesos como
`w ← w − η·∇L`, moviéndose en la dirección que reduce el error según la
pendiente en el punto actual. Cerca de un mínimo, la superficie de la
función de costo se curva, así que un paso demasiado largo (`η` grande) no
se queda cerca del fondo del "tazón": lo cruza de lado a lado y cae en un
punto **más alto** que el de partida. En la siguiente iteración el
gradiente en ese punto es todavía mayor, así que el siguiente paso es aún
más largo — el error se retroalimenta y crece sin control, lo que explica
por qué la explosión ocurre desde el primer chunk y no de forma gradual.

Por el otro extremo, `eta0 = 0.0001` no diverge, pero paga un costo de
velocidad claro: necesita entre 15 y 18 chunks (~20% de la época) para bajar
desde un RMSE inicial que ni siquiera entra en el rango de 150 años del
gráfico hasta estabilizarse en la misma banda de ~9-10 años que `eta0 =
0.001` alcanza en solo 4-5 chunks. En un escenario real donde los datos
llegan como un stream y no se puede reprocesar todo de nuevo, un learning
rate demasiado conservador podría agotar la única pasada disponible sin
haber terminado de converger. En definitiva: el rango de `eta0` que
funciona bien para este problema (con features ya estandarizadas) es
angosto, y `0.001` resulta el mejor compromiso entre velocidad de
convergencia y estabilidad de los cinco valores evaluados.

---

### Ejercicio 2.5 — Learning Rate Schedules

#### 1. Origen del código

Reutiliza la misma infraestructura de los Ejercicios 2.3/2.4:
`X_train_msd` / `y_train_msd` (en memoria), `msd_scaler` (ya ajustado),
`X_test_msd_scaled` / `y_test_msd` (el test set del Ejercicio 2.3) y el
patrón de barajar por época + `partial_fit` por chunk. La tabla de la
Sección 4.3 (`'constant'`, `'optimal'`, `'invscaling'`) es la referencia
para las tres configuraciones a comparar.

#### 2. Qué se añadió / modificó

- Un diccionario `schedule_configs` con los kwargs exactos de cada
  schedule pedidos por el enunciado: `constant` (`eta0=0.001`), `optimal`
  (sin `eta0`, porque ese schedule lo ignora) e `invscaling` (`eta0=0.001,
  power_t=0.25`) — esta forma de parametrizar no existía antes.
- Un bucle que entrena **un modelo nuevo por schedule**, cada uno durante
  `N_EPOCHS_EX25 = 3` épocas, usando un `rng_sched` re-sembrado con la
  misma semilla (`RANDOM_STATE`) al principio de cada modelo — así los tres
  ven exactamente el mismo orden de batches en cada época, aislando el
  efecto del schedule de cualquier diferencia por azar en el orden de los
  datos.
- El diccionario `schedule_val_rmses`, que guarda un RMSE de validación por
  época y por schedule (3 modelos × 3 épocas = 9 valores en total).
- El gráfico con **escala logarítmica en el eje Y** — no estaba planeado
  originalmente, pero fue necesario al descubrir que el schedule `optimal`
  diverge a valores del orden de 10¹²: en escala lineal esa curva aplastaría
  a las otras dos contra el eje X, haciendo imposible comparar `constant`
  vs. `invscaling`.

#### 3. Explicación línea por línea

```python
schedule_configs = {
    'constant':   dict(learning_rate='constant',   eta0=0.001),
    'optimal':    dict(learning_rate='optimal'),
    'invscaling': dict(learning_rate='invscaling', eta0=0.001, power_t=0.25),
}
```
Define los tres schedules exactamente como los describe la tabla de la
Sección 4.3. `eta0` se omite para `'optimal'` porque sklearn lo ignora en
ese modo (usa su propia fórmula basada en `alpha`, el parámetro de
regularización).

```python
for name, kwargs in schedule_configs.items():
    sgd_sched = SGDRegressor(loss='squared_error', random_state=RANDOM_STATE, **kwargs)
    rng_sched = np.random.default_rng(RANDOM_STATE)
    val_rmses = []
```
Por cada schedule se crea un modelo nuevo (pesos en cero) y un generador de
aleatoriedad **reseteado a la misma semilla** — la clave para que los tres
modelos vean idéntico orden de datos en cada época, y así cualquier
diferencia de resultado se deba solo al schedule, no al azar del shuffle.

```python
    for epoch in range(N_EPOCHS_EX25):
        shuffled_idx = rng_sched.permutation(n_train_msd)
        for start in range(0, n_train_msd, CHUNK_SIZE):
            batch_idx = shuffled_idx[start:start + CHUNK_SIZE]
            X_chunk = msd_scaler.transform(X_train_msd[batch_idx])
            y_chunk = y_train_msd[batch_idx]
            sgd_sched.partial_fit(X_chunk, y_chunk)
```
Mismo patrón del Ejercicio 2.3: baraja los índices al inicio de cada época y
entrena chunk a chunk con `partial_fit`, escalando con el `msd_scaler` ya
congelado.

```python
        y_val_pred = sgd_sched.predict(X_test_msd_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_test_msd, y_val_pred))
        val_rmses.append(val_rmse)
```
**Tarea 2**: al cerrar cada época completa, se mide el RMSE sobre el
conjunto de test fijo — igual que en el Ejercicio 2.3, esto es
generalización real, no error de entrenamiento.

```python
ax.set_yscale('log')
```
**Tarea 3** (ajuste necesario): activa escala logarítmica en el eje Y para
poder mostrar en el mismo gráfico un schedule que converge bien (~9-10) y
otro que diverge a ~10¹² sin que uno tape al otro.

#### 4. Resultado obtenido

```
constant    → Val RMSE por época: 9.9274, 10.0818, 10.0053
optimal     → Val RMSE por época: 1031632047538.1833, 573848206953.2244, 21958444618.4444
invscaling  → Val RMSE por época: 9.5400, 9.5282, 9.5141
```

#### 5. Resumen para el análisis

`invscaling` es el claro ganador: baja de forma consistente (9.54 → 9.53 →
9.51 años) y termina por debajo de `constant`, que — igual que en el
Ejercicio 2.3 — se queda oscilando sin mejorar de forma sostenida (9.93 →
10.08 → 10.01). El resultado más interesante, sin embargo, es que
`'optimal'` **diverge por completo** (~10¹² años). No es un error de
implementación: ese schedule calcula su propio tamaño de paso a partir del
parámetro de regularización `alpha`, con una heurística pensada para datos
ya centrados en cero (típico en problemas de clasificación). Nuestro
target real —el año de la canción, con media ≈1998— está lejísimos de
cero, así que el primer paso que calcula `'optimal'` resulta enorme para
esta escala de datos, y el modelo explota desde la primera actualización —
el mismo fenómeno de "paso demasiado largo" del Ejercicio 2.4, solo que
aquí el tamaño del paso lo elige el propio algoritmo en vez de nosotros.

Esto deja dos lecciones. La primera es el trade-off entre `constant` y
schedules decrecientes: un `η_t` que se achica con el tiempo (como
`invscaling`) da pasos grandes al principio para acercarse rápido a una
buena zona y los va acortando después para afinar la puntería sin rebotar
— ideal para un dataset **fijo**, como el de este ejercicio. Pero en un
**stream real con concept drift** (la distribución de los datos cambia con
el tiempo), un `η_t` que ya se volvió minúsculo deja al modelo incapaz de
reaccionar a ese cambio; `constant` nunca deja de poder adaptarse, al costo
de no terminar nunca de asentarse con precisión. La segunda lección es que
ningún schedule es "automáticamente seguro": el buen desempeño de
`'optimal'` en otros contextos depende de supuestos sobre la escala de los
datos (targets centrados en 0) que aquí no se cumplen, y el resultado es
una divergencia tan severa como la de un `eta0` mal elegido a mano.

---
