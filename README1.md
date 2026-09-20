# Análisis final del laboratorio

## 5. Consideraciones del mundo real

### ¿Cuándo usar aprendizaje por lotes y cuándo aprendizaje online?

La elección depende principalmente del tamaño y de la frecuencia con la que llegan los datos.

Si los datos caben en memoria y no cambian constantemente, es conveniente utilizar modelos como Ridge o Lasso dentro de un pipeline. Este enfoque facilita el preprocesamiento, la validación cruzada y la búsqueda de hiperparámetros.

En cambio, cuando el dataset es demasiado grande para cargarlo completo en memoria, el aprendizaje online es una mejor alternativa. En este laboratorio utilizamos `SGDRegressor.partial_fit` para procesar el dataset de canciones por bloques. De esta manera, el modelo actualiza sus parámetros progresivamente sin tener que guardar todas las observaciones al mismo tiempo.

### ¿Por qué es importante escalar las variables?

El escalamiento es especialmente importante para SGD porque el modelo actualiza sus pesos utilizando gradientes. Si una variable tiene valores mucho más grandes que otra, puede dominar las actualizaciones y hacer que el entrenamiento sea inestable.

Por esta razón, primero se ajustó un `StandardScaler` con los datos de entrenamiento y luego se aplicó la misma transformación a cada chunk. También es importante no ajustar el scaler utilizando el conjunto de prueba, porque eso produciría fuga de información.

### ¿Qué ocurre cuando la tasa de aprendizaje es demasiado grande?

Los resultados del ejercicio 2.4 mostraron que las tasas `0.01`, `0.1` y `1.0` divergieron. Esto ocurre porque los cambios en los pesos son demasiado grandes: el modelo salta de un lado a otro de la función de pérdida y no logra acercarse al mínimo.

Por el contrario, las tasas `0.0001` y `0.001` fueron estables. La tasa `0.0001` alcanzó el menor RMSE final entre las tasas estables, con aproximadamente 9.44 años.

### ¿Qué ocurre cuando la tasa es demasiado pequeña?

Una tasa muy pequeña suele ser estable, pero el aprendizaje puede volverse demasiado lento. El modelo necesita muchas actualizaciones y épocas para mejorar. En un sistema real esto puede significar mayor tiempo de entrenamiento y mayor costo computacional.

Por eso no basta con buscar una tasa que no diverja. También se debe verificar que aprenda con suficiente rapidez.

### ¿Por qué es importante la regularización?

La regularización evita que el modelo dependa demasiado de una combinación específica de características. Ridge reduce los coeficientes sin eliminarlos, mientras Lasso puede llevar algunos coeficientes exactamente a cero.

En el ejercicio 1.2, Lasso anuló 7 de 44 coeficientes y Ridge no anuló ninguno. Esto demuestra que Lasso puede hacer selección de características, mientras Ridge conserva toda la información, pero controla el tamaño de los pesos.

### ¿Qué es el concept drift?

El *concept drift* ocurre cuando la relación entre las variables y el objetivo cambia con el tiempo. Por ejemplo, las características musicales que ayudan a predecir el año de una canción podrían comportarse de forma diferente en una nueva generación de canciones.

En estos casos, una tasa constante puede ser útil porque permite que el modelo siga adaptándose a información reciente. Una tasa decreciente puede volverse demasiado pequeña y dejar de reaccionar ante cambios nuevos.

### ¿Por qué evaluar con datos separados?

El error de entrenamiento indica qué tan bien se ajusta el modelo a los datos que ya observó. Sin embargo, no muestra necesariamente cómo funcionará con datos nuevos.

Por eso utilizamos validación cruzada y conjuntos de prueba. En el caso del aprendizaje online, se utilizó el conjunto de prueba oficial del dataset para calcular la `validation RMSE` después de cada época.

### ¿Por qué un pipeline no siempre funciona directamente con `partial_fit`?

Un pipeline tradicional realiza el ajuste completo de sus transformadores antes de entrenar el modelo. En aprendizaje online, los datos llegan por partes y los transformadores también pueden necesitar actualizarse progresivamente.

Por eso, en este laboratorio se administró el `StandardScaler` manualmente mediante `partial_fit` y luego se entregaron al modelo los chunks ya escalados. Esta estrategia es sencilla y permite controlar claramente qué datos se utilizan en cada etapa.

## 6. Resumen y conclusiones principales

Este laboratorio permitió estudiar la regresión lineal desde dos perspectivas: el entrenamiento tradicional con pipelines y el entrenamiento incremental para datasets grandes.

En la primera parte se comprobó que los pipelines ayudan a organizar el flujo completo de machine learning y evitan la fuga de información durante la validación cruzada. También se observó que las transformaciones logarítmicas ayudan a controlar la asimetría de las variables del dataset California Housing.

Las características polinomiales permitieron representar relaciones no lineales. En el ejercicio 1.1, el modelo Ridge de grado 3 obtuvo el mejor resultado entre los grados evaluados, con un `Test RMSE` de 0.5391. Esto indica que el grado 3 capturó mejor la complejidad del problema, aunque un grado mayor no siempre es mejor porque también puede aumentar el riesgo de sobreajuste.

En el ejercicio 1.2 se compararon Ridge y Lasso utilizando características polinomiales de grado 2. Ridge obtuvo un `Test RMSE` de 0.5513, mientras Lasso obtuvo 0.5657. Por lo tanto, Ridge tuvo mejor precisión en esta corrida. Sin embargo, Lasso generó un modelo más disperso al convertir 7 de 44 coeficientes en cero, lo que puede ser útil cuando se busca una mayor interpretabilidad.

En la segunda parte se trabajó con el dataset Year Prediction MSD utilizando aprendizaje online. El modelo procesó 93 chunks y redujo el RMSE desde 44.2414 años en el primer chunk hasta 9.7518 años en el último. Esto demostró que `partial_fit` permite entrenar un modelo sin cargar todo el dataset en memoria.

La curva de convergencia del ejercicio 2.2 mostró una mejora general durante la primera pasada. El RMSE mínimo fue de 7.7227 años en el chunk 53, aunque se observaron variaciones entre bloques debido a las diferencias en la distribución de los datos.

En el ejercicio 2.3 se entrenó el modelo durante 10 épocas. La `validation RMSE` inicial fue 9.9486 y la final fue 9.8289. La mejora fue pequeña y existieron fluctuaciones, por lo que el entrenamiento comenzó a mostrar saturación.

El ejercicio 2.4 mostró que la tasa de aprendizaje es un parámetro crítico. Las tasas `0.01`, `0.1` y `1.0` divergieron, mientras `0.0001` y `0.001` fueron estables. Esto confirma que una tasa alta puede destruir la estabilidad del entrenamiento.

Finalmente, en el ejercicio 2.5 se compararon los calendarios `constant`, `optimal` e `invscaling`. El calendario `invscaling` obtuvo el menor RMSE, con 9.5345 años. El calendario `optimal` produjo valores muy altos en esta configuración, lo que indica que sus parámetros deben ajustarse con cuidado para este dataset.

### Conclusión final

El mejor modelo depende del objetivo del proyecto. Si se busca la menor tasa de error en la primera parte, Ridge de grado 3 fue la mejor alternativa. Si se necesita un modelo más sencillo de interpretar, Lasso puede ser preferible porque elimina características poco importantes.

Para datasets grandes o flujos continuos de información, `SGDRegressor.partial_fit` es una solución práctica y eficiente. Sin embargo, su rendimiento depende mucho del escalamiento, la tasa de aprendizaje, la regularización y el calendario utilizado.

En conclusión, no existe un único modelo que sea siempre el mejor. La decisión debe considerar la precisión, la estabilidad, el tamaño de los datos, el costo computacional y la facilidad para explicar los resultados.
