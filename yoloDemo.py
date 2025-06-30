import os, shutil, random

source_dir = "dataset"
output_dir = "dataset_yolo"
classes = ['car', 'truck']
split_ratio = 0.8  # 80% train, 20% val

for split in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

for class_id, cls in enumerate(classes):
    files = [f for f in os.listdir(f"{source_dir}/{cls}") if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    for i, f in enumerate(files):
        split = 'train' if i < split_index else 'val'
        img_src = f"{source_dir}/{cls}/{f}"
        img_dst = f"{output_dir}/images/{split}/{cls}_{f}"
        label_dst = f"{output_dir}/labels/{split}/{cls}_{f.rsplit('.',1)[0]}.txt"

        shutil.copy(img_src, img_dst)

        # fake bbox: full image box [class x_center y_center width height]
        label = f"{class_id} 0.5 0.5 1.0 1.0\n"  # whole image = object
        with open(label_dst, 'w') as lbl_file:
            lbl_file.write(label)
