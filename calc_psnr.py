import argparse
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_psnr(video1_path, video2_path, output_curve_path=None):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two videos
    and optionally plots the per-frame PSNR curve.

    Args:
        video1_path (str): Path to the first video file (baseline).
        video2_path (str): Path to the second video file (target).
        output_curve_path (str, optional): Path to save the PSNR curve plot.
                                            Defaults to None.

    Returns:
        float: The average PSNR value across all frames with finite PSNR.
               Returns -1.0 if there's an error.
    """
    # Open video captures
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Check if videos opened successfully
    if not video1.isOpened():
        print(f"Error: Could not open baseline video at {video1_path}")
        return -1.0
    if not video2.isOpened():
        print(f"Error: Could not open target video at {video2_path}")
        return -1.0

    # Get video properties
    frame_count1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Verify that video properties match
    if frame_count1 != frame_count2:
        print("Warning: Videos have different frame counts.")
        print(f"Baseline: {frame_count1} frames, Target: {frame_count2} frames.")
    if (width1, height1) != (width2, height2):
        print("Error: Videos have different dimensions.")
        return -1.0

    psnr_per_frame = []
    frame_number = 0

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # Break the loop if either video has ended
        if not ret1 or not ret2:
            break

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)

        # If MSE is 0, the frames are identical, PSNR is infinite
        if mse == 0:
            psnr = float("inf")
        else:
            max_pixel_value = 255.0
            psnr = 10 * math.log10((max_pixel_value**2) / mse)
        
        print(f"Frame {frame_number + 1}: PSNR = {psnr if psnr == float('inf') else f'{psnr:.4f} dB'}")
        psnr_per_frame.append(psnr)
        frame_number += 1

    # Release video captures
    video1.release()
    video2.release()

    if frame_number == 0:
        return 0.0

    # Calculate average PSNR, excluding infinite values
    finite_psnr_values = [p for p in psnr_per_frame if p != float('inf')]
    average_psnr = sum(finite_psnr_values) / len(finite_psnr_values) if finite_psnr_values else float('inf')

    # Plot and save the PSNR curve if a path is provided
    if output_curve_path:
        plt.figure(figsize=(10, 5))
        # Use a high value for plotting 'inf' for better visualization
        plot_psnr_values = [100 if p == float('inf') else p for p in psnr_per_frame]
        plt.plot(range(frame_number), plot_psnr_values)
        plt.title('PSNR per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        # Indicate where PSNR was infinite
        inf_frames = [i for i, p in enumerate(psnr_per_frame) if p == float('inf')]
        if inf_frames:
             plt.plot(inf_frames, [100] * len(inf_frames), 'ro', label='Infinite PSNR (Identical Frame)')
             plt.legend()
        
        try:
            plt.savefig(output_curve_path)
            print(f"\nPSNR curve saved to {output_curve_path}")
        except Exception as e:
            print(f"\nError saving plot: {e}")
        plt.close()


    return average_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PSNR between two videos.")
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to the baseline video."
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Path to the target video."
    )
    parser.add_argument(
        "--output_curve_path", 
        type=str, 
        default=None, 
        help="Optional: Path to save the PSNR curve plot image (e.g., 'psnr_curve.png')."
    )
    args = parser.parse_args()

    avg_psnr = calculate_psnr(args.baseline, args.target, args.output_curve_path)

    if avg_psnr != -1.0:
        print("\n-------------------------------------")
        print(f"Average PSNR (finite frames): {avg_psnr if avg_psnr != float('inf') else 'Infinite'} dB")
        print("-------------------------------------")
