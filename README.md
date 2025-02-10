# AR_Draft_TestFiles

Este repositorio contiene una serie de pruebas realizadas en el contexto de la evaluacion del paper  Drafting in Collectible Card Games via Reinforcement Learning utilizando aprendizaje por refuerzo.  
Los experimentos fueron realizados con el entorno [gym-locm](https://github.com/ronaldosvieira/gym-locm), basado en OpenAI Gym, para evaluar distintos enfoques de selección de cartas en la fase de draft.

## Archivos del Repositorio

### Prueba de Entorno (`Prueba_de_entorno.py`)
**Objetivo:**  
Verificar que el entorno `gym-locm` está correctamente instalado y que la simulación de partidas se ejecuta sin errores.  

**Funcionalidad:**  
- Carga el entorno `LOCM-draft-v0`.  
- Ejecuta partidas de draft utilizando estrategias heurísticas (`Random` y `Max-Attack`).  
- Evalúa tasas de victoria para confirmar el correcto funcionamiento del entorno.

### Extracción de Features (`Extraccion_de_features.py`)
**Objetivo:**  
Analizar la representación de las cartas y los estados en el draft, asegurando que los modelos de aprendizaje por refuerzo reciban la información correctamente.  

**Funcionalidad:**  
- Utiliza las funciones `encode_card()` y `encode_state_draft()` para transformar cartas y estados en vectores numéricos.  
- Verifica la normalización de los datos y la correcta dimensión de las representaciones (`16` para cartas individuales, `48` para estados de draft).  

### Predictor Modificado (`Predictor_modificado.py`)
**Objetivo:**  
Comparar el rendimiento de distintos modelos de draft entrenados, solucionando problemas detectados en el script original `predictor.py`.  

**Mejoras Implementadas:**  
- Eliminación de procesos externos: el `battle agent` ahora se ejecuta internamente en lugar de depender de procesos remotos.  
- Corrección en el manejo de argumentos: se añadieron `--draft-1` , `--draft-2` , `approach-1` y `approach-2` para especificar claramente los modelos a comparar.  
- Mayor información en la salida: reporta tasas de victoria y estadísticas detalladas sobre la selección de cartas.  

**Ejemplo de uso:**  
```bash
python3 predictor_modificado.py \
    --draft-1 Modelos/Inmediate_Optimized/30000.zip \
    --draft-2 Modelos/History/30000.zip \
    --approach-1 immediate \
    --approach-2 history
