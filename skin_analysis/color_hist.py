import matplotlib.pyplot as plt
import csv
import numpy as np

color_dir = './color_results'

# Define the dataset information
datasets = [
    {'name': 'CelebAMask-HQ', 'csv_file': f'{color_dir}/CelebA.csv'},
    {'name': 'FFHQ', 'csv_file': f'{color_dir}/FFHQ.csv'},
    {'name': 'test_dataset', 'csv_file': f'{color_dir}/test_dataset.csv'},
    {'name': 'VGG_Face2', 'csv_file': f'{color_dir}/VGG_Face2.csv'},
]

def read_data_from_csv(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            hue = float(row['hue'])
            lum = float(row['lum'])
            data.append((hue, lum))
    return data

def plot_lum_data(lum_data, title, xlabel, ylabel, filename):
    lum = np.array(lum_data)
    light_mask = lum >= 60
    dark_mask = lum < 60

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lum[light_mask], bins=20, range=(0, 100), 
            alpha=0.7, color='orange', label='light')
    ax.hist(lum[dark_mask], bins=20, range=(0, 100), 
            alpha=0.7, color='saddlebrown', label='dark')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_hue_data(hue_data, title, xlabel, ylabel, filename):
    hue = np.array(hue_data)
    yellow_mask = hue >= 55
    red_mask = hue < 55

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(hue[red_mask], bins=20, range=(0, 80), 
            alpha=0.7, color='red', label='red')
    ax.hist(hue[yellow_mask], bins=20, range=(0, 80), 
            alpha=0.7, color='yellow', label='yellow')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot data for each dataset and each type
for dataset in datasets:
    data = read_data_from_csv(dataset['csv_file'])
    hue, lum = zip(*data)
    hue = np.array(hue)
    lum = np.array(lum)

    # Plot lum histogram
    title_lum = f"{dataset['name']} - Perceptual Lightness (L*)"
    filename_lum = f"./color_results/{dataset['name']}_lum_hist.png"
    plot_lum_data(lum, title_lum, 'perceptual lightness L *', 'Count', filename_lum)

    # Plot hue histogram
    title_hue = f"{dataset['name']} - Hue Angle (h*)"
    filename_hue = f"./color_results/{dataset['name']}_hue_hist.png"
    plot_hue_data(hue, title_hue, 'hue angle h *', 'Count', filename_hue)

print('\nDone!')