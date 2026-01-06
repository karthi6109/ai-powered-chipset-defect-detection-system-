import argparse
from pathlib import Path
import yaml
import shutil
import os

def find_label_file(img_path):
    # Try same folder with .txt, or sibling 'labels' folder
    p = Path(img_path)
    txt_same = p.with_suffix('.txt')
    if txt_same.exists():
        return txt_same
    # sibling labels directory
    parent = p.parent
    if parent.name == 'images':
        labels_dir = parent.parent / 'labels'
    else:
        labels_dir = parent / 'labels'
    candidate = labels_dir / p.name.replace(p.suffix, '.txt')
    if candidate.exists():
        return candidate
    return None

def process_split(images_dir, out_split_dir):
    images_dir = Path(images_dir)
    out_split_dir = Path(out_split_dir)
    out_defect = out_split_dir / 'defect'
    out_normal = out_split_dir / 'normal'
    out_defect.mkdir(parents=True, exist_ok=True)
    out_normal.mkdir(parents=True, exist_ok=True)

    exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
    img_files = []
    for e in exts:
        img_files.extend(images_dir.rglob(e))

    defect_count = 0
    normal_count = 0
    for img in img_files:
        lbl = find_label_file(img)
        is_defect = False
        if lbl and lbl.exists():
            try:
                txt = lbl.read_text().strip()
                if txt:
                    is_defect = True
            except Exception:
                is_defect = True

        target = out_defect if is_defect else out_normal
        shutil.copy2(img, target / img.name)
        if is_defect:
            defect_count += 1
        else:
            normal_count += 1

    return {'defect': defect_count, 'normal': normal_count, 'total': defect_count+normal_count}

def main(dataset_root, out_root='dataset'):
    dataset_root = Path(dataset_root).resolve()
    with open(dataset_root / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_key in ('train','val','test'):
        if split_key in data:
            rel = data[split_key]
            # try multiple resolution strategies
            candidates = []
            try_paths = [dataset_root / rel, dataset_root.parent / rel, Path(rel)]
            images_path = None
            for tp in try_paths:
                tp = tp.resolve()
                if tp.exists():
                    images_path = tp
                    break

            # fallback: search for directories containing many images under dataset_root
            if images_path is None:
                exts = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
                best = None
                best_count = 0
                for p in dataset_root.rglob('*'):
                    if not p.is_dir():
                        continue
                    count = 0
                    for e in exts:
                        for _ in p.glob(e):
                            count += 1
                            if count > best_count:
                                break
                    if count > best_count:
                        best = p
                        best_count = count
                if best is not None and best_count>0:
                    images_path = best

            if images_path is None or not images_path.exists():
                print(f"Warning: images path for {split_key} not found (tried {try_paths}).")
                continue
            out_dir = out_root / split_key
            stats = process_split(images_path, out_dir)
            summary[split_key] = stats

    print('Conversion complete. Summary:')
    for k,v in summary.items():
        print(f" {k}: {v['total']} images ({v['defect']} defect, {v['normal']} normal)")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', help='Path to YOLO dataset root containing data.yaml')
    p.add_argument('--out', default='dataset', help='Output folder for converted dataset')
    args = p.parse_args()
    main(args.root, args.out)
