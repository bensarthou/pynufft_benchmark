#!/bin/bash
# Indique au système que l'argument qui suit est le programme utilisé pour exécuter ce fichier
# En règle générale, les "#" servent à mettre en commentaire le texte qui suit comme ici
echo Comparaison PyNufft/pynfft
set -e
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 512 -j 4 -o samples_2D.npy -t U_2D_512_4_g_a -g True -a True
python3 -W ignore test_pynfft.py -i mri_img_2D.npy -o samples_2D.npy -t F_2D_c_a -a True
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 512 -j 4 -o samples_2D.npy -t U_2D_512_4_c_a -g False -a True

python3 -W ignore test_NFFT.py U_2D_512_4_g_a.npy U_2D_512_4_c_a.npy F_2D_c_a.npy
