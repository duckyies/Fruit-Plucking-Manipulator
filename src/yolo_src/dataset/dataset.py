import os
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from tqdm import tqdm

maindir = os.path.dirname(__file__)
parquetdir = r"C:\Users\TARA\Desktop\Fruit-Picking-Manipulator-With-Computer-Vision\src\yolo_src\dataset"
parqfiles = {
    "train": os.path.join(parquetdir, "train-00000-of-00001.parquet"),
    "validation": os.path.join(parquetdir, "test-00000-of-00001.parquet"),
}

outputdir = os.path.dirname(__file__)

def process_parquet(split, parquet_path):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    parquet = pq.ParquetFile(parquet_path)
    index = 0

    for batch in parquet.iter_batches(batch_size=16):
        table = batch.to_pydict()
        images = table["image"]
        labels = table["label"]

        for img_data, label in zip(images, labels):
            if label not in [0, 2]:
                continue

            if label == 0:
                class_name = "unripe"
            elif label == 2:
                class_name = "ripe"

            save_dir = os.path.join(outputdir, split, class_name)
            os.makedirs(save_dir, exist_ok=True)

            image = Image.open(BytesIO(img_data["bytes"]))
            image.save(os.path.join(save_dir, f"{index}.jpg"))

            index += 1

    print(f"{split}: banana-only ({index} images extracted)")

for split, path in parqfiles.items():
    process_parquet(split, path)
    