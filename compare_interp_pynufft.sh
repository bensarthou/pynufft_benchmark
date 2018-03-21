#!/bin/bash
# Indique au système que l'argument qui suit est le programme utilisé pour exécuter ce fichier
# En règle générale, les "#" servent à mettre en commentaire le texte qui suit comme ici
echo Comparaison Taille interpolateur Pynufft
set -e
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 1024 -j 2 -o samples_2D.npy -t U_2D_1024_2_c_a -g False -a True
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 1024 -j 4 -o samples_2D.npy -t U_2D_1024_4_c_a -g False -a True
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 1024 -j 6 -o samples_2D.npy -t U_2D_1024_6_c_a -g False -a True
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 1024 -j 8 -o samples_2D.npy -t U_2D_1024_8_c_a -g False -a True
python3 -W ignore test_pynufft.py -i mri_img_2D.npy -k 1024 -j 10 -o samples_2D.npy -t U_2D_1024_10_c_a -g False -a True

python3 -W ignore test_NFFT.py U_2D_1024_2_c_a.npy U_2D_1024_4_c_a.npy U_2D_1024_6_c_a.npy U_2D_1024_8_c_a.npy U_2D_1024_10_c_a.npy
