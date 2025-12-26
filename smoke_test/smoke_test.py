#!/usr/bin/env python3
"""
OpenVLA Smoke Test: Multi-sample inference script with BridgeDataV2.

Loads openvla-7b model and runs inference on samples from BridgeDataV2 dataset.
Generates comparison plots and GIFs for each episode.

Usage:
    python smoke_test/smoke_test.py
    python smoke_test/smoke_test.py --max_steps 20
    python smoke_test/smoke_test.py --output_dir ./my_outputs
"""

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION - Add/remove tfrecord files here
# ============================================================================
TFRECORD_SAMPLES = [
    "/home/sapan-alienware/datasets/rlds/bridge_orig/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024",
    "/home/sapan-alienware/datasets/rlds/bridge_orig/1.0.0/bridge_dataset-train.tfrecord-00001-of-01024",
    # Add more samples here:
    # "/path/to/bridge_dataset-train.tfrecord-00002-of-01024",
]

# Base URL for downloading missing samples
TFRECORD_BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
# ============================================================================


def download_if_missing(tfrecord_path: str) -> bool:
    """Download tfrecord if it doesn't exist."""
    if os.path.exists(tfrecord_path):
        return True
    
    filename = os.path.basename(tfrecord_path)
    url = TFRECORD_BASE_URL + filename
    
    print(f"      Downloading {filename}...")
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    
    import urllib.request
    try:
        urllib.request.urlretrieve(url, tfrecord_path)
        print(f"      Downloaded successfully!")
        return True
    except Exception as e:
        print(f"      Failed to download: {e}")
        return False


def load_episode(tfrecord_path: str, max_steps: int = 20):
    """Load full episode from BridgeDataV2 tfrecord file."""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    
    feature_description = {
        'steps/observation/image_0': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/action': tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
        'steps/language_instruction': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    }
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in raw_dataset.take(1):
        example = tf.io.parse_single_example(raw_record, feature_description)
        images_raw = example['steps/observation/image_0']
        actions_raw = example['steps/action']
        instructions = example['steps/language_instruction']
        
        instruction = instructions[0].numpy().decode('utf-8') if len(instructions) > 0 else "manipulate object"
        num_steps = min(max_steps, len(images_raw))
        
        images = []
        gt_actions = []
        for i in range(num_steps):
            img = tf.io.decode_jpeg(images_raw[i])
            images.append(Image.fromarray(img.numpy()))
            gt_actions.append(actions_raw[i].numpy())
        
        return images, np.array(gt_actions), instruction, num_steps
    
    return None, None, None, 0


def run_inference(model, processor, images, instruction, device):
    """Run inference on all images in episode."""
    pred_actions = []
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    
    for image in images:
        inputs = processor(prompt, image.convert("RGB")).to(device, dtype=torch.bfloat16)
        if 'attention_mask' in inputs:
            del inputs['attention_mask']
        
        with torch.no_grad():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        pred_actions.append(action)
    
    return np.array(pred_actions)


def generate_plots(images, gt_actions, pred_actions, instruction, output_dir, episode_name):
    """Generate comparison plots and GIF for an episode."""
    num_steps = len(images)
    
    # Compute cumulative EE positions from action deltas
    gt_positions = np.cumsum(gt_actions[:, :3], axis=0)
    pred_positions = np.cumsum(pred_actions[:, :3], axis=0)
    
    # ========== 1. Static comparison plot ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Episode: '{instruction}'\n{episode_name}", fontsize=12)
    
    # Position components
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        axes[0, i].plot(gt_actions[:, i], f'{color}-', label='Ground Truth', linewidth=2)
        axes[0, i].plot(pred_actions[:, i], f'{color}--', label='Predicted', linewidth=2)
        axes[0, i].set_xlabel('Step')
        axes[0, i].set_ylabel(f'{label} Delta (m)')
        axes[0, i].set_title(f'Position {label}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Rotation components
    for i, (label, color) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], ['r', 'g', 'b'])):
        axes[1, i].plot(gt_actions[:, i+3], f'{color}-', label='Ground Truth', linewidth=2)
        axes[1, i].plot(pred_actions[:, i+3], f'{color}--', label='Predicted', linewidth=2)
        axes[1, i].set_xlabel('Step')
        axes[1, i].set_ylabel(f'{label} Delta (rad)')
        axes[1, i].set_title(f'Rotation {label}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150)
    plt.close()
    
    # ========== 2. 3D Trajectory plot ==========
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
            'r--', linewidth=2, label='Predicted')
    
    ax.scatter(*gt_positions[0], c='green', s=150, marker='*', label='Start')
    ax.scatter(*gt_positions[-1], c='blue', s=100, marker='o', label='GT End')
    ax.scatter(*pred_positions[-1], c='red', s=100, marker='^', label='Pred End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"3D End-Effector Trajectory\n'{instruction}'")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_3d.png', dpi=150)
    plt.close()
    
    # ========== 3. Gripper comparison ==========
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_actions[:, 6], 'b-', label='Ground Truth', linewidth=2, marker='o')
    ax.plot(pred_actions[:, 6], 'r--', label='Predicted', linewidth=2, marker='^')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Open/Close threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gripper Value')
    ax.set_title(f"Gripper Command\n'{instruction}'")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gripper_comparison.png', dpi=150)
    plt.close()
    
    # ========== 4. Animated GIF ==========
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_img = fig.add_subplot(122)
    
    all_pos = np.vstack([gt_positions, pred_positions])
    margin = 0.02
    x_min, x_max = all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin
    y_min, y_max = all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin
    z_min, z_max = all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin
    
    def update(frame):
        ax_3d.clear()
        ax_img.clear()
        
        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])
        
        if frame > 0:
            ax_3d.plot(gt_positions[:frame+1, 0], gt_positions[:frame+1, 1], gt_positions[:frame+1, 2], 
                       'b-', linewidth=2, label='Ground Truth')
            ax_3d.plot(pred_positions[:frame+1, 0], pred_positions[:frame+1, 1], pred_positions[:frame+1, 2], 
                       'r--', linewidth=2, label='Predicted')
        
        ax_3d.scatter(*gt_positions[frame], c='blue', s=100, marker='o', edgecolors='black', zorder=5)
        ax_3d.scatter(*pred_positions[frame], c='red', s=100, marker='^', edgecolors='black', zorder=5)
        ax_3d.scatter(*gt_positions[0], c='green', s=150, marker='*', edgecolors='black', label='Start')
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title(f'End-Effector Trajectory\nStep {frame+1}/{num_steps}')
        ax_3d.legend(loc='upper left')
        ax_3d.view_init(elev=20, azim=45 + frame * 2)
        
        ax_img.imshow(images[frame])
        ax_img.set_title(f"Camera View - Step {frame+1}\n'{instruction}'")
        ax_img.axis('off')
        
        gt_grip = "open" if gt_actions[frame, 6] > 0.5 else "close"
        pred_grip = "open" if pred_actions[frame, 6] > 0.5 else "close"
        info_text = f"GT: [{gt_actions[frame, 0]:.3f}, {gt_actions[frame, 1]:.3f}, {gt_actions[frame, 2]:.3f}] grip:{gt_grip}\n"
        info_text += f"Pred: [{pred_actions[frame, 0]:.3f}, {pred_actions[frame, 1]:.3f}, {pred_actions[frame, 2]:.3f}] grip:{pred_grip}"
        ax_img.text(0.5, -0.1, info_text, transform=ax_img.transAxes, fontsize=9, 
                    ha='center', va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=num_steps, interval=800, blit=False)
    anim.save(output_dir / 'trajectory.gif', writer='pillow', fps=1.5)
    plt.close()
    
    # ========== 5. Save statistics ==========
    stats = {
        'episode_name': episode_name,
        'instruction': instruction,
        'num_steps': num_steps,
        'position_errors': [np.linalg.norm(pred_actions[i, :3] - gt_actions[i, :3]) for i in range(num_steps)],
        'rotation_errors': [np.linalg.norm(pred_actions[i, 3:6] - gt_actions[i, 3:6]) for i in range(num_steps)],
        'total_errors': [np.linalg.norm(pred_actions[i] - gt_actions[i]) for i in range(num_steps)],
    }
    stats['mean_position_error'] = np.mean(stats['position_errors'])
    stats['mean_rotation_error'] = np.mean(stats['rotation_errors'])
    stats['mean_total_error'] = np.mean(stats['total_errors'])
    
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write(f"Episode: {episode_name}\n")
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Number of steps: {num_steps}\n")
        f.write(f"\nMean Position Error: {stats['mean_position_error']:.4f}\n")
        f.write(f"Mean Rotation Error: {stats['mean_rotation_error']:.4f}\n")
        f.write(f"Mean Total Error: {stats['mean_total_error']:.4f}\n")
        f.write(f"\nPer-step errors:\n")
        for i in range(num_steps):
            f.write(f"  Step {i+1}: pos={stats['position_errors'][i]:.4f}, rot={stats['rotation_errors'][i]:.4f}, total={stats['total_errors'][i]:.4f}\n")
    
    return stats


def main(output_dir: str = None, max_steps: int = 20):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    print("=" * 70)
    print("OpenVLA Smoke Test - Multi-Sample BridgeDataV2 Inference")
    print("=" * 70)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Number of samples: {len(TFRECORD_SAMPLES)}")
    
    # Filter to existing/downloadable samples
    valid_samples = []
    print("\n[1/3] Checking samples...")
    for tfrecord_path in TFRECORD_SAMPLES:
        if download_if_missing(tfrecord_path):
            valid_samples.append(tfrecord_path)
        else:
            print(f"      Skipping: {tfrecord_path}")
    
    if not valid_samples:
        print("ERROR: No valid samples found!")
        return
    
    print(f"      Found {len(valid_samples)} valid samples")
    
    # Load model once
    print("\n[2/3] Loading OpenVLA model (openvla/openvla-7b)...")
    processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        'openvla/openvla-7b',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    print("      Model loaded!")
    
    # Process each sample
    print("\n[3/3] Processing episodes...")
    all_stats = []
    
    for idx, tfrecord_path in enumerate(valid_samples):
        episode_name = os.path.basename(tfrecord_path).replace('.tfrecord', '').replace('bridge_dataset-train.', '')
        episode_dir = output_dir / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Episode {idx+1}/{len(valid_samples)}: {episode_name}")
        print('='*70)
        
        # Load episode
        print("  Loading episode...")
        images, gt_actions, instruction, num_steps = load_episode(tfrecord_path, max_steps)
        if images is None:
            print("  ERROR: Failed to load episode")
            continue
        print(f"  Instruction: '{instruction}'")
        print(f"  Steps: {num_steps}")
        
        # Run inference
        print("  Running inference...")
        pred_actions = run_inference(model, processor, images, instruction, device)
        print(f"  Inference complete!")
        
        # Generate plots
        print("  Generating plots and GIF...")
        stats = generate_plots(images, gt_actions, pred_actions, instruction, episode_dir, episode_name)
        all_stats.append(stats)
        
        print(f"  Mean Position Error: {stats['mean_position_error']:.4f}")
        print(f"  Mean Total Error: {stats['mean_total_error']:.4f}")
        print(f"  Outputs saved to: {episode_dir}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for stats in all_stats:
        print(f"\n{stats['episode_name']}:")
        print(f"  Instruction: '{stats['instruction']}'")
        print(f"  Mean Position Error: {stats['mean_position_error']:.4f}")
        print(f"  Mean Total Error: {stats['mean_total_error']:.4f}")
    
    if all_stats:
        overall_pos_err = np.mean([s['mean_position_error'] for s in all_stats])
        overall_total_err = np.mean([s['mean_total_error'] for s in all_stats])
        print(f"\nOverall Mean Position Error: {overall_pos_err:.4f}")
        print(f"Overall Mean Total Error: {overall_total_err:.4f}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("=" * 70)
    print("Smoke test completed!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenVLA Multi-Sample Smoke Test")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots and GIFs")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per episode")
    args = parser.parse_args()
    
    main(output_dir=args.output_dir, max_steps=args.max_steps)
