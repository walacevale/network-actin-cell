import os  # Biblioteca padrão
import cv2  # Biblioteca de terceiros
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.filters import sato, threshold_li  # Importações específicas
from skimage.morphology import medial_axis

def calculate_skeleton_and_medial_axis(img: np.ndarray):
    """
    Calcula o esqueleto e a distância media de uma imagem binária.

    Args:
        img (np.ndarray): Imagem binária de entrada.

    Returns:
        tuple[np.ndarray, np.ndarray]: Esqueleto da imagem e distâncias para o fundo.
    """
    skel, distance = medial_axis(img, return_distance=True)
    dist_on_skel = distance * skel
    skel = (skel > 0).astype(np.uint8) * 255
    return skel, dist_on_skel

def calculate_distance(x_grid: np.ndarray, y_grid: np.ndarray, x: int, y: int):
    """
    Calcula a distância de cada pixel para um ponto específico.

    Args:
        x_grid (np.ndarray): Grid de coordenadas x.
        y_grid (np.ndarray): Grid de coordenadas y.
        x (int): Coordenada x do ponto de referência.
        y (int): Coordenada y do ponto de referência.

    Returns:
        np.ndarray: Matriz de distâncias.
    """
    return np.sqrt((x_grid - x)**2 + (y_grid - y)**2)

def sum_pixels_in_circle(image: np.ndarray, x: int, y: int, radius: int):
    """
    Soma os valores dos pixels dentro de um círculo de raio específico.

    Args:
        image (np.ndarray): Imagem de entrada.
        x (int): Coordenada x do centro do círculo.
        y (int): Coordenada y do centro do círculo.
        radius (int): Raio do círculo.

    Returns:
        int: Soma dos valores dos pixels dentro do círculo.
    """
    height, width = image.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    distances = calculate_distance(x_grid, y_grid, x, y)
    circle_mask = distances <= radius
    pixels_in_circle = image[circle_mask]
    total_sum = np.sum(pixels_in_circle)
    return total_sum

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_image_name(image_path):
    return os.path.basename(image_path).split('_')[-1].split('.')[0]

def load_images_from_folder(folder_path: str):
    """
    Carrega todas as imagens em escala de cinza e 16 bits de uma pasta.

    Args:
        folder_path (str): Caminho para a pasta contendo as imagens.

    Returns:
        tuple: Uma tupla contendo listas de imagens em escala de cinza, 
               imagens em 16 bits e nomes das imagens.
    """
    gray_images = []
    images_16bit = []
    image_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_img)
            images_16bit.append(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH))
            image_names.append(get_image_name(img_path))
    
    return gray_images, images_16bit, image_names

def apply_sato_filter_and_threshold(image: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro Sato e um limiar de Li para binarizar a imagem.

    Args:
        image (np.ndarray): Imagem de entrada em escala de cinza.

    Returns:
        np.ndarray: Imagem binarizada.
    """
    sato_img = sato(image, sigmas=[2], black_ridges=False, mode='reflect', cval=0)
    thresh = threshold_li(sato_img)
    binary = (sato_img > thresh).astype(np.uint8) * 255
    return binary

def calculate_skeleton_and_medial_axis(binary_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o esqueleto e o eixo medial de uma imagem binarizada.

    Args:
        binary_image (np.ndarray): Imagem binarizada.

    Returns:
        tuple: Uma tupla contendo o esqueleto e a distância ao fundo para pixels do esqueleto.
    """
    skeleton, distance = medial_axis(binary_image, return_distance=True)
    dist_on_skel = distance * skeleton
    skeleton = (skeleton > 0).astype(np.uint8) * 255
    return skeleton, dist_on_skel

def sum_pixels_in_circle(image: np.ndarray, x: int, y: int, radius: float) -> float:
    """
    Soma os valores dos pixels dentro de um círculo de raio específico.

    Args:
        image (np.ndarray): Imagem de entrada.
        x (int): Coordenada x do centro do círculo.
        y (int): Coordenada y do centro do círculo.
        radius (float): Raio do círculo.

    Returns:
        float: Soma dos valores dos pixels dentro do círculo.
    """
    height, width = image.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
    circle_mask = distances <= radius
    pixels_in_circle = image[circle_mask]
    return np.sum(pixels_in_circle)

def process_image(image: np.ndarray, image_16bit: np.ndarray, name: str, output_path: str):
    """
    Processa uma imagem aplicando filtro Sato, limiar de Li, esqueleto e soma de intensidades.

    Args:
        image (np.ndarray): Imagem em escala de cinza.
        image_16bit (np.ndarray): Imagem em 16 bits.
        name (str): Nome da imagem.
        output_path (str): Caminho para salvar a imagem processada.
    """
    binary_image = apply_sato_filter_and_threshold(image)
    skeleton, skeleton_medial = calculate_skeleton_and_medial_axis(binary_image)

    position_skeleton = np.where(skeleton > 0)
    number_position = len(position_skeleton[1])
    new_skeleton_intense = np.zeros_like(image, dtype=np.float32)

    for i in range(number_position):
        radius = skeleton_medial[position_skeleton[0][i], position_skeleton[1][i]]
        soma_pixels = sum_pixels_in_circle(image_16bit, position_skeleton[1][i], position_skeleton[0][i], radius)
        new_skeleton_intense[position_skeleton[0][i], position_skeleton[1][i]] = soma_pixels

    output_file_path = os.path.join(output_path, f"{name}.tiff")
    cv2.imwrite(output_file_path, new_skeleton_intense)

def skeleton_intense(path_to_folder: str, output_path: str):
    """
    Processa todas as imagens em uma pasta para calcular a intensidade do esqueleto.

    Args:
        path_to_folder (str): Caminho para a pasta contendo as imagens.
        output_path (str): Caminho para salvar as imagens processadas.
    """
    gray_images, images_16bit, image_names = load_images_from_folder(path_to_folder)

    for img, img_16bit, img_name in tqdm(zip(gray_images, images_16bit, image_names), total=len(gray_images)):
        process_image(img, img_16bit, img_name, output_path)