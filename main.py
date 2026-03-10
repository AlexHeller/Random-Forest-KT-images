import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

ROOT_DIR = "Data"
IMG_SIZE = 128
RANDOM_STATE = 42


def extract_features(image):
    glcm = graycomatrix(image, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    ASM = graycoprops(glcm, 'ASM').mean()

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    mean_val = np.mean(image)
    std_val = np.std(image)
    max_val = np.max(image)

    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM, mean_val, std_val, max_val])
    features = np.concatenate((features, lbp_hist))

    return features


def load_all_data(root_path):
    print(f"Сканирование папки '{root_path}'...")
    X = []
    y = []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    folder_name = os.path.basename(root).lower()

                    if "normal" in folder_name:
                        label = "Normal"
                    elif any(cancer_type in folder_name for cancer_type in ['adenocarcinoma', 'large', 'squamous']):
                        label = "Cancer"
                    else:
                        continue

                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.medianBlur(img, 3)

                    feats = extract_features(img)

                    X.append(feats)
                    y.append(label)

                except Exception as e:
                    pass

    print(f"Завершено. Всего загружено: {len(X)} снимков.")
    return np.array(X), np.array(y)


# --- ЗАПУСК ---
if __name__ == "__main__":
    if not os.path.exists(ROOT_DIR):
        print(f"ОШИБКА: Папка '{ROOT_DIR}' не найдена рядом со скриптом.")
    else:
        start_time = time.time()

        # 1. Загрузка
        X, y = load_all_data(ROOT_DIR)

        if len(X) == 0:
            print("Не найдено изображений. Проверьте структуру папок.")
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = le.classes_
            print(f"Классы для скрининга: {class_names}")

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=RANDOM_STATE,
                                                                stratify=y_encoded)

            n_estimators_list = list(range(10, 201, 10))

            train_accuracies = []
            test_accuracies = []
            train_f1_scores = []
            test_f1_scores = []

            best_model = None
            best_f1 = 0
            best_n_trees = 0

            for n in n_estimators_list:
                clf = RandomForestClassifier(n_estimators=n,
                                             max_depth=None,
                                             random_state=RANDOM_STATE,
                                             n_jobs=-1)
                clf.fit(X_train, y_train)

                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)


                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                train_f1_scores.append(train_f1)
                test_f1_scores.append(test_f1)

                print(f"Деревьев: {n:3d} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")


                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_model = clf
                    best_n_trees = n

            print(f"\nЛучшая модель использует {best_n_trees} деревьев (Test F1 = {best_f1:.4f}).")

