#!/bin/bash
# Indique au système que l'argument qui suit est le programme utilisé pour exécuter ce fichier
# En règle générale, les "#" servent à mettre en commentaire le texte qui suit comme ici
echo Comparaison PyNufft/pynfft
set -e
python3 -W ignore test_pynufft.py -k 512 -j 4 -o samples_sparkling.npy -t U_new_spark -g False
python3 -W ignore test_pynfft.py -o samples_sparkling.npy -t F_new_spark

python3 -W ignore test_NFFT.py U_new_spark.npy F_new_spark.npy
