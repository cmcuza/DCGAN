Fichero del proyecto Deep Convolucional Generative Adversarial Network

El fichero tiene un ejemplo completo del uso de las GAN con Transformadores Espaciales y las funciones de costo modificadas para mejorar los resultados.

Data: CelebA (descargar y poner dentro de la carpeta data)

Para entrenar:

python main.py --dataset celebA --train

Para comprobar los resultados:

python main.py --dataset celebA --test 

Para predecir sobre un conjunto nuevo:

python main.py --dataset celebA --test --predict list.txt

El fichero stn_example es de muestra para las STN, funciona con la mnist_cluttered_60x60_6distortions.npz


