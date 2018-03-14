#!/bin/bash
# Indique au système que l'argument qui suit est le programme utilisé pour exécuter ce fichier
# En règle générale, les "#" servent à mettre en commentaire le texte qui suit comme ici
echo Comparaison Taille interpolateur Pynufft
set -e
python3 -W ignore test_pynufft.py -k 512 -j 2 -o om_pynufft.npy -t U_512_2_c -g False
python3 -W ignore test_pynufft.py -k 512 -j 4 -o om_pynufft.npy -t U_512_4_c -g False
python3 -W ignore test_pynufft.py -k 512 -j 6 -o om_pynufft.npy -t U_512_6_c -g False
python3 -W ignore test_pynufft.py -k 512 -j 8 -o om_pynufft.npy -t U_512_8_c -g False
python3 -W ignore test_pynufft.py -k 512 -j 10 -o om_pynufft.npy -t U_512_10_c -g False

python3 -W ignore test_NFFT.py U_512_2_c.npy U_512_4_c.npy U_512_6_c.npy U_512_8_c.npy U_512_10_c.npy
