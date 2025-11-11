"""
Extract VQGAN weights from Transformer checkpoint
"""
import torch
import sys

def extract_vqgan_from_transformer(ckpt_path, output_path):
    """
    Extract first_stage_model (VQGAN) weights from Net2NetTransformer checkpoint
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Print all keys to understand structure
    print("\n=== Checkpoint Keys ===")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Total keys: {len(state_dict)}")

        # Find VQGAN keys
        vqgan_keys = [k for k in state_dict.keys() if k.startswith('first_stage_model.')]
        print(f"VQGAN keys: {len(vqgan_keys)}")

        if vqgan_keys:
            # Extract VQGAN weights
            vqgan_state_dict = {}
            for k in vqgan_keys:
                # Remove 'first_stage_model.' prefix
                new_key = k.replace('first_stage_model.', '')
                vqgan_state_dict[new_key] = state_dict[k]

            # Create new checkpoint with only VQGAN
            new_checkpoint = {
                'state_dict': vqgan_state_dict,
                'epoch': checkpoint.get('epoch', 0),
                'global_step': checkpoint.get('global_step', 0),
            }

            print(f"\n=== Extracted VQGAN State Dict ===")
            print(f"Keys: {len(vqgan_state_dict)}")
            print("\nSample keys:")
            for i, k in enumerate(list(vqgan_state_dict.keys())[:10]):
                print(f"  {k}")

            # Save
            torch.save(new_checkpoint, output_path)
            print(f"\n✅ VQGAN weights saved to: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
            return True
        else:
            print("❌ No VQGAN keys found in checkpoint!")
            print("\nAvailable key prefixes:")
            prefixes = set([k.split('.')[0] for k in state_dict.keys()])
            for p in sorted(prefixes):
                print(f"  - {p}")
            return False
    else:
        print("❌ No 'state_dict' found in checkpoint!")
        print(f"Available keys: {list(checkpoint.keys())}")
        return False

if __name__ == "__main__":
    import os

    ckpt_path = "/Users/thachha/Desktop/AIO2025-official/AIMA/CP-WSSS/taming-transformers/logs/2020-11-13T21-41-45_faceshq_transformer 2/checkpoints/last.ckpt"
    output_path = "/Users/thachha/Desktop/AIO2025-official/AIMA/CP-WSSS/taming-transformers/logs/vqgan_faceshq_extracted.ckpt"

    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    success = extract_vqgan_from_transformer(ckpt_path, output_path)
    sys.exit(0 if success else 1)
