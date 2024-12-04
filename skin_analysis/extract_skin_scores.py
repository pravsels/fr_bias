import os
import csv
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2lab, lab2rgb
from sklearn import cluster
from skimage.io import imread
from tqdm import tqdm 
import multiprocessing as mp
import argparse

def read_img(fpath):
    img = Image.open(fpath).convert('RGB')
    return img

def get_hue(a_values, b_values, eps=1e-8):
    """Compute hue angle"""
    return np.degrees(np.arctan(b_values / (a_values + eps)))

def mode_hist(x, bins='sturges'):
    """Compute a histogram and return the mode"""
    hist, bins = np.histogram(x, bins=bins)
    mode = bins[hist.argmax()]
    return mode

def clustering(x, n_clusters=5, random_state=2021):
    model = cluster.KMeans(n_clusters, random_state=random_state, n_init='auto')
    model.fit(x)
    return model.labels_, model

def get_scalar_values(skin_smoothed_lab, labels, topk=3, bins='sturges'):
    # gather values of interest
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    skin_smoothed = lab2rgb(skin_smoothed_lab)

    # concatenate data to be clustered (L, h, and RGB for visualization)
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle,
                                 skin_smoothed[:, 0], skin_smoothed[:, 1], skin_smoothed[:, 2]]).T

    # Extract skin pixels for each mask (by clusters)
    n_clusters = len(np.unique(labels))
    masked_skin = [data_to_cluster[labels == i, :] for i in range(n_clusters)]
    n_pixels = np.asarray([np.sum(labels == i) for i in range(n_clusters)])

    # get scalar values per cluster
    keys = ['lum', 'hue', 'red', 'green', 'blue']
    res = {}
    for i, key in enumerate(keys):
        res[key] = np.array([mode_hist(part[:, i], bins=bins)
                             for part in masked_skin])

    # only keep top3 in luminance and avarage results
    idx = np.argsort(res['lum'])[::-1][:topk]
    total = np.sum(n_pixels[idx])
    res_topk = {}
    for key in keys:
        res_topk[key] = np.average(res[key][idx], weights=n_pixels[idx])
        res_topk[key+'_std'] = np.sqrt(np.average((res[key][idx]-res_topk[key])**2, weights=n_pixels[idx]))

    return res_topk

def get_skin_values(img, mask, n_clusters=5):
    # smoothing
    img_smoothed = gaussian(img,
                            sigma=(1, 1),
                            truncate=4,
                            channel_axis=-1)

    # get skin pixels (shape will be Mx3) and go to Lab
    skin_smoothed = img_smoothed[mask]
    skin_smoothed_lab = rgb2lab(skin_smoothed)

    res = {}

    # L and hue
    hue_angle = get_hue(skin_smoothed_lab[:, 1], skin_smoothed_lab[:, 2])
    data_to_cluster = np.vstack([skin_smoothed_lab[:, 0], hue_angle]).T
    labels, model = clustering(data_to_cluster, n_clusters=n_clusters)
    tmp = get_scalar_values(skin_smoothed_lab, labels)
    res['lum'] = tmp['lum']
    res['hue'] = tmp['hue']
    res['lum_std'] = tmp['lum_std']
    res['hue_std'] = tmp['hue_std']

    # also extract RGB for visualization purposes
    res['red'] = tmp['red']
    res['green'] = tmp['green']
    res['blue'] = tmp['blue']
    res['red_std'] = tmp['red_std']
    res['green_std'] = tmp['green_std']
    res['blue_std'] = tmp['blue_std']

    return res

def find_image_files(root_dir):
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.png') and not filename.endswith('_mask.png'):
                image_files.append(os.path.relpath(os.path.join(dirpath, filename), root_dir))

    return image_files

def process_image(args):
    image_file, dataset_root, masks_dir = args
    
    # Find the corresponding mask file
    image_path = os.path.join(dataset_root, image_file)
    mask_file = os.path.splitext(image_file)[0] + '_mask.png'
    mask_path = os.path.join(masks_dir, mask_file)

    # Check if the mask file exists
    if not os.path.exists(mask_path):
        print(f"Mask file not found for image: {image_file}. Skipping.")
        print(f'Mask path : {mask_path}')
        return None

    img_original = imread(image_path)
    mask = imread(mask_path)

    # get values
    tmp = get_skin_values(np.asarray(img_original), np.asarray(mask) == 1)

    # Return the results as a dictionary
    return {
        'filename': image_file,
        'lum': tmp['lum'], 'lum_std': tmp['lum_std'],
        'hue': tmp['hue'], 'hue_std': tmp['hue_std'],
        'red': tmp['red'], 'red_std': tmp['red_std'],
        'green': tmp['green'], 'green_std': tmp['green_std'],
        'blue': tmp['blue'], 'blue_std': tmp['blue_std']
    }

def main():
    parser = argparse.ArgumentParser(description='Extract skin scores from images')
    parser.add_argument('dir', type=str,
                        help='Folder to process (ST1, ST2, etc)')
    
    args = parser.parse_args()
    
    dataset_root = f'../synthpar2/{args.dir}'
    masks_dir = os.path.join('../synthpar2/masks', args.dir)
    
    print('Dataset root:', dataset_root)
    print('Masks directory:', masks_dir)

    # Get a list of image files recursively
    image_files = find_image_files(dataset_root)

    # Create color_results folder
    color_results_dir = './color_results'
    os.makedirs(color_results_dir, exist_ok=True)
    # Create a CSV file to store the results    
    csv_file = os.path.join(color_results_dir, f'{args.dir}.csv')
    fieldnames = ['filename', 
                  'lum', 'lum_std', 
                  'hue', 'hue_std', 
                  'red', 'red_std', 
                  'green', 'green_std', 
                  'blue', 'blue_std']

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        pool = mp.Pool(processes=35)

        # Process images in parallel
        args_list = [(image_file, dataset_root, masks_dir) for image_file in image_files]
        results = list(tqdm(pool.imap(process_image, args_list), 
                            total=len(image_files), 
                            desc='processing images...'))

        for result in results:
            if result is not None:
                writer.writerow(result)

        # Close the pool
        pool.close()
        pool.join()

    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()

