# Tarea 2 — CIT3352: Set-Union Knapsack Problem (SUKP)

Solución completa en C++ con cuatro algoritmos y un script de graficación en Python.

## Algoritmos implementados

1. **Greedy determinista** — agrega el ítem factible con mejor razón beneficio/costo marginal (determinista, misma solución siempre).
2. **Greedy estocástico (GRASP)** — construcción con lista restringida de candidatos (RCL); reproducible por semilla.
3. **Simulated Annealing** (trayectoria) — opera sobre un decodificador (repara+rellena); T0 calibrada por el método de Ben-Ameur, enfriamiento geométrico. Se ejecuta desde inicio greedy determinista y desde greedy estocástico.
4. **Algoritmo Genético memético** (población) — población inicial sembrada con greedy estocástico, selección por torneo, cruce uniforme, mutación bit-flip, elitismo y decodificación de cada hijo.

## Estructura

```
codigo/        sukp.cpp  (todos los algoritmos)
               graficos.py  (genera figuras y tablas)
resultados/    *.csv  (summary, convergencia, barridos, tablas)
figuras/       *.png  (convergencia, comparaciones, boxplots, barridos)
```

## Compilación y ejecución (C++)

```bash
g++ -O3 -march=native -o sukp codigo/sukp.cpp

# Uso:  ./sukp <archivo_instancia> <nombre> <dir_salida> [RUNS] [--sweep]
mkdir -p results
for inst in easy medium1 medium2 hard; do
    ./sukp ruta/$inst.txt $inst results 10
done

# Barridos de parámetros (solo se necesita una instancia representativa):
./sukp ruta/medium1.txt medium1 results 10 --sweep
```

Esto genera los CSV en `results/`.

## Graficación (Python)

Requiere `matplotlib`, `numpy`, `pandas`.

```bash
python3 codigo/graficos.py        # lee results/ y escribe results/figs/ + tablas
```

## Formato de instancia

```
m n ne B            # m alternativas, n recursos, ne incidencias, B capacidad
p_1 ... p_m         # beneficios de las alternativas
w_1 ... w_n         # pesos de los recursos
i j                 # ne líneas: la alternativa i usa el recurso j (índices base 0)
...
```

## Reproducibilidad

Cada ejecución usa semilla fija (`run` = 0..RUNS-1) para greedy estocástico, SA y GA, de modo que los resultados son reproducibles. El greedy determinista produce siempre la misma solución.
