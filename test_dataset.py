import os

# Directories to check
start_sequences_dir = r"/data/scratch/ec23984/cobot_data/start_sequences"
stop_sequences_dir = r"/data/scratch/ec23984/cobot_data/stop_sequences"

# File extensions to consider as images
valid_extensions = ('.jpg', '.jpeg', '.png')

def find_incomplete_sequences(directory, required_images=15):
    """Find directories with fewer than required_images."""
    incomplete_dirs = []

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            sequence_path = os.path.join(root, dir_name)
            # Count valid image files in the directory
            image_count = len([
                file for file in os.listdir(sequence_path)
                if file.lower().endswith(valid_extensions)
            ])
            if image_count < required_images:
                incomplete_dirs.append((sequence_path, image_count))

    return incomplete_dirs

# Check both directories
incomplete_start = find_incomplete_sequences(start_sequences_dir)
incomplete_stop = find_incomplete_sequences(stop_sequences_dir)

# Print results
print("\n[Incomplete Start Sequences]")
for path, count in incomplete_start:
    print(f"{path} - {count} images")

print("\n[Incomplete Stop Sequences]")
for path, count in incomplete_stop:
    print(f"{path} - {count} images")

# Summary
print("\nSummary:")
print(f"Total incomplete 'start' sequences: {len(incomplete_start)}")
print(f"Total incomplete 'stop' sequences: {len(incomplete_stop)}")
