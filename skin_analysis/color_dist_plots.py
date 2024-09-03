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

lum_boundary = 60
hue_boundary = 55 

def read_data_from_csv(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            hue = float(row['hue'])
            lum = float(row['lum'])
            data.append((hue, lum))
    return data

def plot_data(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    hue, lum = zip(*data)

    hue = np.array([h for h in hue])
    lum = np.array(lum)

    hue_norm = hue / 360.0

    # Create a colormap based on the hue values
    cmap = plt.cm.hsv
    colors = cmap(hue_norm)

    ax.scatter(hue, lum, s=10, c=colors, 
               alpha=0.7, marker='h')

    ax.axhline(y=lum_boundary, linestyle='--', color='blue', linewidth=1.5)
    ax.axvline(x=hue_boundary, linestyle='--', color='blue', linewidth=1.5)
            

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig

# Plot data for each dataset
for dataset in datasets:
    data = read_data_from_csv(dataset['csv_file'])
    title = f"({chr(97 + datasets.index(dataset))}) {dataset['name']}"
    fig = plot_data(data, title, 'hue angle h *', 'perceptual lightness L *')
    filename = f"./color_results/{dataset['name']}.png" 
    fig.savefig(filename)

print('showing plots!')

