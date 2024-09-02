import csv
import numpy as np

color_dir = './color_results'

# Define the dataset information
datasets = [
    {'name': 'CelebAMask-HQ', 'csv_file': f'{color_dir}/CelebA.csv'},
    {'name': 'FFHQ', 'csv_file': f'{color_dir}/FFHQ.csv'},
    {'name': 'SynthPar', 'csv_file': f'{color_dir}/SynthPar.csv'},
    {'name': 'VGG_Face2', 'csv_file': f'{color_dir}/VGG_Face2.csv'},
    {'name': '80k', 'csv_file': f'{color_dir}/80k.csv'},
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

def calculate_percentages(data):
    hue, lum = zip(*data)
    hue = np.array(hue)
    lum = np.array(lum)

    yellow = np.sum(hue > hue_boundary)
    red = np.sum(hue <= hue_boundary)

    light = np.sum(lum > lum_boundary)
    dark = np.sum(lum <= lum_boundary)

    total = len(hue)

    percentages = {
        'Red': red / total * 100,
        'Yellow': yellow / total * 100,
        'Light': light / total * 100,
        'Dark': dark / total * 100,
        'Total': 100
    }

    return percentages

# Calculate percentages for each dataset
for dataset in datasets:
    data = read_data_from_csv(dataset['csv_file'])

    percentages = calculate_percentages(data)
    print(f"\nSkin tone percentages for {dataset['name']}:")
    print("{:<8} {:<10} {:<10} {:<10}".format('Hue/Tone', 'Light', 'Dark', 'Total'))
    print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f}".format('Red', percentages['Red'] * percentages['Light'] / 100, percentages['Red'] * percentages['Dark'] / 100, percentages['Red']))
    print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f}".format('Yellow', percentages['Yellow'] * percentages['Light'] / 100, percentages['Yellow'] * percentages['Dark'] / 100, percentages['Yellow']))
    print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f}".format('Total', percentages['Light'], percentages['Dark'], percentages['Total']))

print('\nDone!')

