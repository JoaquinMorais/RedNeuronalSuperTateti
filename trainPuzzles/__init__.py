"""Entrenamiento por puzzles tácticos de Super Tateti ("mate en N").

Método independiente del resto del repositorio: genera posiciones con victoria
forzada en exactamente N jugadas propias y entrena ``SuperTatetiNet`` por
descenso de gradiente, avanzando de nivel cuando la red acierta un umbral.

Ver ``README.md`` en esta carpeta para instrucciones de ejecución.

``evolve_puzzles`` ofrece una alternativa evolutiva memética (gradiente + mutación
+ selección) que mantiene una población de redes y se queda con la de mayor
acierto; su configuración vive en ``evo_config``.
"""

__all__ = ["puzzles", "train", "evo_config", "evolve_puzzles"]
__version__ = "1.1.0"
