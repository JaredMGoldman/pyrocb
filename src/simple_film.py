import cv2
import os
import pandas as pd

from pft_gen_parallel import plot_spatiotemporal_data

if __name__ == "__main__":
    # Create sorted list of image paths
    pft_dir = '/home/jaredgoldman/dev/pyrocb/outputs/plots/pfts'
    data_path = f'{pft_dir}/pft_vals.csv'
    image_folder = f'{pft_dir}/GFS'
    os.makedirs(image_folder, exist_ok=True)
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # plot_spatiotemporal_data(df, output_dir=image_folder, save_df = False)
    video_name = os.path.join(image_folder,'GFS_2026_0525_48hr_PFT.mp4')
    out_ims = image_folder 
    images = sorted([img for img in os.listdir(out_ims) if img.endswith(".png")])

    # Get dimensions from first image

    frame = cv2.imread(os.path.join(out_ims, images[0]))
    h, w, l = frame.shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 24, (w, h))

    for image in images:
        video.write(cv2.imread(os.path.join(out_ims, image)))

    video.release()