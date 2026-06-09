import os
import csv
DATASET_PATH = "archive"
CSV_FILE = "dataset_labels.csv"
id_counter = 1
with open(CSV_FILE , mode="w" , newline="" , encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "IMAGE", "EXPRESSION"])
    for expression in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, expression)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_path = os.path.join(expression, image_name)
                    writer.writerow([id_counter ,image_path ,expression ])
                    id_counter += 1
print(f"CSV created: {CSV_FILE}")