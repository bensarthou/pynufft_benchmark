#!/bin/bash
# Indique au système que l'argument qui suit est le programme utilisé pour exécuter ce fichier
# En règle générale, les "#" servent à mettre en commentaire le texte qui suit comme ici
echo Comparaison PyNufft/pynfft
set -e
python3 -W ignore test_pynufft.py -k 512 -j 4 -o om_pynufft.npy -t U_512_4_c_a -g False -a True
python3 -W ignore test_pynfft.py -o om_pynufft.npy -t F_c_a -a True

python3 -W ignore test_NFFT.py U_512_4_c_a.npy F_c_a.npy
