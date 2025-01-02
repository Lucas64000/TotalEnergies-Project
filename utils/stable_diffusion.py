import random
import json
from diffusers import AutoPipelineForText2Image
import torch

def load_pipeline_diffuser()
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    return pipeline_text2image

def save_keywords_to_json(keywords, filename='data/keywords.json'):
    with open(filename, 'w') as f:
        json.dump(keywords, f, indent=4)

def load_keywords(file_path='data/keywords.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_prompt(keywords):
    time_of_day = random.choice(keywords['time_of_day'])
    weather = random.choice(keywords['weather'])
    landscape = random.choice(keywords['landscape'])
    temperature = random.choice(keywords['temperature'])
    return (f"Realistic {landscape} wild environment during {time_of_day} with no animals, "
            f"{weather} weather, {temperature} temperature, as if captured by a wildlife camera but without animals.")

def generate_images(keywords, pipeline, start=0, end=1000):
    from tqdm import tqdm
    for i in tqdm(range(start, end)):
        prompt = generate_prompt(keywords)
        image = pipeline(prompt=prompt).images[0]
        save_image(image, save_folder="data/background/", filename=f"generated_image_{i+1}.png")


import random
import json 
def load_keywords(file_path='keywords.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

# Fonction pour générer un prompt en fonction des variables
def generate_prompt(keywords):
    time_of_day = random.choice(keywords['time_of_day'])
    weather = random.choice(keywords['weather'])
    landscape = random.choice(keywords['landscape'])
    temperature = random.choice(keywords['temperature'])
    
    prompt = f"Realistic {landscape} wild environment during {time_of_day} with no animals, {weather} weather, {temperature} temperature, as if captured by a wildlife camera but without animals."

    return prompt

# Variables possibles
keywords = {
    "time_of_day": ['morning', 'afternoon', 'evening', 'night'],
    "weather": [
        'clear', 'cloudy', 'rainy', 'stormy', 'foggy', 
        'sunny', 'snowy', 'windy'
    ],
    "landscape": [
        'moutain range'
    ],
    "temperature": [
        'cold', 'moderate', 'warm'
    ],
}

with open('keywords.json', 'w') as f:
    json.dump(keywords, f, indent=4)

keywords = load_keywords()

import os
from tqdm import tqdm
from IPython.display import clear_output

# Fonction pour sauvegarder une image
def save_image(image, save_folder, filename):
    # Crée le dossier s'il n'existe pas
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # Construit le chemin du fichier
    filepath = os.path.join(save_folder, filename)
    # Sauvegarde l'image
    image.save(filepath)
    print(f"Image saved at {filepath}")

def generate_images(keywords, pipeline, start=0, end=1000):

    for i in tqdm(range(start, end)):
        if (i + 1) % 5 == 0:
            clear_output(wait=True)
        prompt = generate_prompt(keywords)
        print(f"Generated Prompt {i+1}: {prompt}")
        
        image = pipeline(prompt=prompt).images[0]
        
        # Sauvegarde l'image dans le dossier "background"
        save_image(image, save_folder="background/", filename=f"generated_image_{i+1}.png")
        plt.figure(figsize=(8, 8))  # Ajuster la taille si nécessaire
        plt.title(f"Generated Image {i+1}")
        plt.imshow(image)
        plt.axis('off')  # Supprimer les axes pour une meilleure présentation
        plt.show()


def generate_images2(keywords, pipeline, tab_indexes):

    for i in tqdm(tab_indexes):
        if (i + 1) % 5 == 0:
            clear_output(wait=False)
        prompt = generate_prompt(keywords)
        print(f"Generated Prompt {i}: {prompt}")
        
        image = pipeline(prompt=prompt).images[0]
        
        # Sauvegarde l'image dans le dossier "background"
        save_image(image, save_folder="background/", filename=f"generated_image_{i}.png")
        plt.figure(figsize=(8, 8))  # Ajuster la taille si nécessaire
        plt.title(f"Generated Image {i}")
        plt.imshow(image)
        plt.axis('off')  # Supprimer les axes pour une meilleure présentation
        plt.show()


def generate_image(prompt, pipeline):
    image = pipeline(prompt=prompt).images[0]
    plt.figure(figsize=(8, 8))  # Ajuster la taille si nécessaire
    plt.imshow(image)
    plt.axis('off')  # Supprimer les axes pour une meilleure présentation
    plt.show()
    return image