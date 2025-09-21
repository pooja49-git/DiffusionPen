import os
import argparse
from tqdm import tqdm

def create_annotation_file(data_dir, output_file):
    """
    Scans a directory and creates an annotation file.
    Assumes filenames are the transcriptions (e.g., 'नमस्ते.jpg').
    """
    print(f"Scanning directory: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        return

    annotations = []
    for dirpath, _, filenames in tqdm(os.walk(data_dir), desc=f"Scanning {os.path.basename(data_dir)}"):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(dirpath, filename)
                transcription = os.path.splitext(filename)[0]
                annotations.append(f"{full_path},{transcription}")

    if not annotations:
        print("No images found.")
        return

    print(f"Writing {len(annotations)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in annotations:
            f.write(line + '\n')
            
    print(f"Done! Created {output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an annotation file from an image directory.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the root of the dataset directory (e.g., .../HindiSeg/train).')
    parser.add_argument('--output_file', type=str, required=True, help='Name of the output annotation file (e.g., train_annotations.txt).')
    
    args = parser.parse_args()
    create_annotation_file(args.data_dir, args.output_file)