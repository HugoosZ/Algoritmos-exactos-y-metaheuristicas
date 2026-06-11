#!/bin/bash
# Barrido experimental BSG-RCL — Entrega 2 CIT3352
# Requiere los archivos thpack{1,7,11,15}.txt (OR-Library) en este directorio.
# Presupuesto aprox: 40 instancias x 30 s x 7 corridas ~ 2,3 horas.
T=30; CSV=resultados_finales.csv
for SET in 1 7 11 15; do
  F="thpack${SET}.txt"
  [ -f "$F" ] || { echo "FALTA $F — descargar de OR-Library"; continue; }
  MF=100; [ "$SET" -ge 8 ] && MF=98          # bloques generales en BR8-BR15
  echo "== Set BR${SET} (min_fr=$MF) =="
  ./bsg_rcl "$F" --problems 1-10 --time $T --gamma 1.0 --seed 1 --min_fr $MF --csv $CSV 2>/dev/null
  for G in 2.0 3.0; do
    for S in 1 2 3; do
      ./bsg_rcl "$F" --problems 1-10 --time $T --gamma $G --seed $S --min_fr $MF --csv $CSV 2>/dev/null
    done
  done
done
echo "Listo. Resultados en $CSV"
