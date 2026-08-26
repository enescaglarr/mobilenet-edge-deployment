"""Build a small ImageNet-val subset for Colab (run on the Mac).

    python make_subset.py --src ../dataset --dst dataset_subset --per-class 2
    zip -r dataset_subset.zip dataset_subset

2 images/class -> 2000 images, ~80 MB. Enough for a stable top-1/top-5 estimate
(±1%) and it uploads to Colab in a minute.
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../dataset")
    ap.add_argument("--dst", default="dataset_subset")
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    n = 0
    for c in classes:
        files = sorted(f for f in os.listdir(os.path.join(args.src, c)) if f.lower().endswith(".jpg"))
        os.makedirs(os.path.join(args.dst, c), exist_ok=True)
        for f in files[:args.per_class]:
            shutil.copy2(os.path.join(args.src, c, f), os.path.join(args.dst, c, f))
            n += 1
    print(f"{n} images from {len(classes)} classes -> {args.dst}")


if __name__ == "__main__":
    main()
