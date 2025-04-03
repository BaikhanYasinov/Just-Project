import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Функция для загрузки изображения
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


# Инициализация модели
detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
detection_model.eval()

# Загрузка изображения и преобразование в тензор
image_path = '32.jpg'
image = load_image(image_path)
image_tensor = F.to_tensor(image).unsqueeze(0)

# Получение предсказания наличия объектов
with torch.no_grad():
    predictions = detection_model(image_tensor)

boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Порог уверенности и отбор предсказаний
threshold = 0.75
selected_indices = [i for i, score in enumerate(scores) if score > threshold]

# Загрузка семантических меток классов
with open("coco_classes_all.txt", "r") as f:
    coco_classes = [line.strip() for line in f.readlines()]

# Отображение результатов предсказаний
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Список ситуаций для распознавания
situations = [['dining table', 'bottle'], ['person', 'fork'], ['cell phone', 'person']]
detected_classes = set()




for number, i in enumerate(selected_indices):
    box = boxes[i].numpy()
    label = labels[i].item()
    score = scores[i].item()
    detected_class = coco_classes[label]

    detected_classes.add(detected_class)

    rect_color = 'green'


    rect = mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              linewidth=2, edgecolor=rect_color, facecolor='none')
    ax.add_patch(rect)

    ax.text(box[0], box[1] - 10, f"{number + 1}: {detected_class} ({score:.2f})", color='red', fontsize=12)
    print(f"Объект: {number + 1}, Класс: {detected_class}, Вероятность: {score:.2f}")

# Выдлеение пересеччений

    def check_intersection(box1, box2):
        return (box1[0] < box2[2] and box1[2] > box2[0] and
                box1[1] < box2[3] and box1[3] > box2[1])

    for j in selected_indices:
        if i != j and check_intersection(box, boxes[j]):
            for situation in situations:
                if (coco_classes[labels[i]] == situation[0] and coco_classes[labels[j]] == situation[1]
                        or coco_classes[labels[i]] == situation[1] and coco_classes[labels[j]] == situation[0]):
                    rect.set_edgecolor('red')
                    rect.set_linewidth(4)
                    break


plt.axis('off')
plt.show()
