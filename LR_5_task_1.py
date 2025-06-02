import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Візуалізація класифікатора
def visualize_classifier(classifier, X, y, title, filename=None):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks(np.arange(int(min_x), int(max_x), 1.0))
    plt.yticks(np.arange(int(min_y), int(max_y), 1.0))
    plt.title(title)
    plt.show()

# Парсер аргументів
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Класифікація даних за допомогою ансамблевого навчання')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True,
                        choices=['rf', 'erf'], help="Тип класифікатора: 'rf' або 'erf'")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Завантаження даних
    input_file = "DRF.txt"

    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)

    # Розділення даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    # Вибір моделі
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    classifier = RandomForestClassifier(**params) if classifier_type == 'rf' else ExtraTreesClassifier(**params)

    # Навчання
    classifier.fit(X_train, y_train)

    # Візуалізація
    visualize_classifier(classifier, X_train, y_train, 'Training dataset', 'train_dataset.png')
    visualize_classifier(classifier, X_test, y_test, 'Тестовий набір даних', 'test_dataset.png')

    # Оцінка
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#" * 40)
    print("Класифікація на тренувальному наборі:\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))

    print("Класифікація на тестовому наборі:\n")
    print(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
    print("#" * 40)

    # Оцінка довіри
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    print("\nРівні довірливості:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))
        print('\nТочка:', datapoint)
        print('Клас:', predicted_class)
        print('Ймовірності:', probabilities)

    # Візуалізація тестових точок
    visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 'Test datapoints', 'test_points.png')
