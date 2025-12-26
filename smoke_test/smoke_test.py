#!/usr/bin/env python3
"""
OpenVLA Smoke Test: Single-sample inference script with BridgeDataV2.

Loads openvla-7b model and runs inference on a sample from BridgeDataV2 dataset.

Usage:
    python smoke_test/smoke_test.py
    python smoke_test/smoke_test.py --tfrecord /path/to/tfrecord --step 10
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_bridgev2_sample(tfrecord_path: str, step_idx: int = 5):
    """Load a sample from BridgeDataV2 tfrecord file."""
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
        
        # Get image
        images = example['steps/observation/image_0']
        num_steps = len(images)
        step_idx = min(step_idx, num_steps - 1)
        img = tf.io.decode_jpeg(images[step_idx])
        image = Image.fromarray(img.numpy())
        
        # Get instruction
        instructions = example['steps/language_instruction']
        instruction = instructions[0].numpy().decode('utf-8') if len(instructions) > 0 else "pick up the object"
        
        # Get ground truth action
        actions = example['steps/action']
        gt_action = actions[step_idx].numpy() if len(actions) > step_idx else actions[0].numpy()
        
        return image, instruction, gt_action, num_steps


def create_dummy_sample():
    """Create a dummy sample for testing without dataset."""
    image = Image.new('RGB', (256, 256), color='gray')
    instruction = "pick up the object"
    gt_action = None
    return image, instruction, gt_action, 1


def main(tfrecord_path: str = None, step_idx: int = 5):
    print("=" * 60)
    print("OpenVLA Smoke Test - BridgeDataV2 Inference")
    print("=" * 60)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load sample
    print("\n[1/4] Loading sample...")
    if tfrecord_path and os.path.exists(tfrecord_path):
        print(f"      Loading from: {tfrecord_path}")
        image, instruction, gt_action, num_steps = load_bridgev2_sample(tfrecord_path, step_idx)
        print(f"      Episode has {num_steps} steps, using step {step_idx}")
    else:
        print("      Using dummy sample (no tfrecord provided)")
        image, instruction, gt_action, _ = create_dummy_sample()
    
    print(f"      Image size: {image.size}")
    print(f"      Instruction: '{instruction}'")
    if gt_action is not None:
        print(f"      Ground truth action: {gt_action}")
    
    # Load the processor and model
    print("\n[2/4] Loading OpenVLA model (openvla/openvla-7b)...")
    processor = AutoProcessor.from_pretrained(
        'openvla/openvla-7b',
        trust_remote_code=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        'openvla/openvla-7b',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    
    print("      Model loaded successfully!")
    
    # Format prompt using the VLA prompt template
    print("\n[3/4] Preparing input...")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    print(f"      Prompt: '{prompt[:80]}...'")
    
    # Prepare inputs
    inputs = processor(prompt, image.convert("RGB")).to(device, dtype=torch.bfloat16)
    
    # Remove attention_mask to avoid shape mismatch issue
    if 'attention_mask' in inputs:
        del inputs['attention_mask']
    
    # Run inference
    print("\n[4/4] Running inference...")
    with torch.no_grad():
        predicted_action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nInstruction: '{instruction}'")
    
    print(f"\nPredicted Action (7-DoF):")
    print(f"  {predicted_action}")
    print(f"  - Position delta (x, y, z): {predicted_action[:3]}")
    print(f"  - Rotation delta (roll, pitch, yaw): {predicted_action[3:6]}")
    print(f"  - Gripper: {predicted_action[6]:.3f} ({'open' if predicted_action[6] > 0.5 else 'close'})")
    
    if gt_action is not None:
        print(f"\nGround Truth Action (7-DoF):")
        print(f"  {gt_action}")
        print(f"  - Position delta (x, y, z): {gt_action[:3]}")
        print(f"  - Rotation delta (roll, pitch, yaw): {gt_action[3:6]}")
        print(f"  - Gripper: {gt_action[6]:.3f} ({'open' if gt_action[6] > 0.5 else 'close'})")
        print(f"\nAction Error (L2 norm): {np.linalg.norm(predicted_action - gt_action):.4f}")
    
    print("\n" + "=" * 60)
    print("Smoke test completed successfully!")
    print("=" * 60)
    
    return predicted_action


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenVLA Smoke Test")
    parser.add_argument("--tfrecord", type=str, default=None, help="Path to BridgeDataV2 tfrecord file")
    parser.add_argument("--step", type=int, default=5, help="Step index within episode")
    args = parser.parse_args()
    
    # Default to downloaded tfrecord if available
    default_tfrecord = "/home/sapan-alienware/datasets/rlds/bridge_orig/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024"
    tfrecord_path = args.tfrecord or (default_tfrecord if os.path.exists(default_tfrecord) else None)
    
    main(tfrecord_path=tfrecord_path, step_idx=args.step)
