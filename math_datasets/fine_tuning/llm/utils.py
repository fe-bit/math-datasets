import os
import re

def get_latest_checkpoint_dir(output_dir):
    checkpoint_dirs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and re.match(r"checkpoint-\d+", d)
    ]

    if not checkpoint_dirs:
        return None

    # Extract the checkpoint number and find the highest one
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(output_dir, latest_checkpoint)
