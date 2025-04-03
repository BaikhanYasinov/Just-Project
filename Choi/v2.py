import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision import transforms, models
import torch.nn.functional as torch_func
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def read_file_to_array_readlines(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def browse_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выбери изображение", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
    )
    return file_path


def handle_process():
    first_obj = situations_combo_1.get()
    second_obj = situations_combo_2.get()
    situations = [(first_obj, second_obj)]
    visualize(image_path_var.get(), situations)


def check_intersection(box1, box2):
    return (box1[0] < box2[2] and box1[2] > box2[0] and
            box1[1] < box2[3] and box1[3] > box2[1])


def visualize(image_path, situations):
    pic = Image.open(image_path)

    image_tensor_2 = F.to_tensor(pic).unsqueeze(0)

    print(image_tensor_2)

    detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    detection_model.eval()

    with torch.no_grad():
        predictions = detection_model(image_tensor_2)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']


    threshold = 0.75
    selected_indices = [i for i, score in enumerate(scores) if score > threshold]

    with open("coco_classes_all.txt", "r") as f:
        coco_classes = [line.strip() for line in f.readlines()]


    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(pic)
    number = 1
    for i in selected_indices:
        box = boxes[i].numpy()
        label = labels[i].item()
        score = scores[i].item()
        detected_class = coco_classes[label]

        rect = mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                  linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1]-10, f"{number}", color='red', fontsize=12)
        print(f"Объект: {number},  Класс: {label} {detected_class}, Вероятность: {score:.2f}")
        number += 1

        for j in selected_indices:
            if i != j and check_intersection(box, boxes[j]):
                for situation in situations:
                    if coco_classes[labels[i]] == situation[0] and coco_classes[labels[j]] == situation[1] or coco_classes[labels[i]] == situation[1] and coco_classes[labels[j]] == situation[0]:
                        rect.set_edgecolor('red')
                        rect.set_linewidth(4)
                        break

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    sit = read_file_to_array_readlines('coco_classes_all.txt')
    sit.sort()
    root = tk.Tk()
    root.title("Детекция опасных ситуаций")
    image_path_var = tk.StringVar()
    tk.Button(root,bg='#98FB98', text="Выберите изображение", command=lambda: image_path_var.set(browse_image())).pack(pady=5)
    situations_var_1 = tk.StringVar()
    situations_label_1 = ttk.Label(root, text="Первый объект").pack(pady=2)
    situations_combo_1 = ttk.Combobox(root, textvariable=situations_var_1,
                                    values=sit, state='readonly')
    situations_combo_1.pack(pady=5)


    situations_var_2 = tk.StringVar()
    situations_label = ttk.Label(root, text="Второй объект").pack(pady=2)
    situations_combo_2 = ttk.Combobox(root, textvariable=situations_var_2,
                                    values=sit, state='readonly')
    situations_combo_2.pack(pady=5)
    tk.Button(root, bg ='#FA8072', text="Выполнить программу", command=handle_process).pack(pady=10)
    root.mainloop()
