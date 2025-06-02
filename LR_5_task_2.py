import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Завантаження вхідних даних
input_file = "DI.txt"
data = np.loadtxt (input_file, delimiter=',')
X, y = data[:, :- 1], data[:, -1]

# Поділ вхідних даних на два класи на підставі міток
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

# Візуалізація вхідних даних
plt.figure ()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, 
            facecolors='black', edgecolors='black',
            linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
            facecolors='white', edgecolors='black',
            linewidth=1, marker='o')
plt.title('Вхідні дані')

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Класифікатор на основі гранично випадкових лісів
params = {'n_estimators': 100, 'max_depth': 4,'random_state': 0}
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params = {'n_estimators': 100, 'max_depth': 4,
                  'random_state': 0, 'class_weight': 'balanced'}

    else:
        raise TypeError ("Invalid input argument; should be 'balance'")

classifier = ExtraTreesClassifier ( ** params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Навчальний набір даних')

y_test_pred = classifier.predict (X_test)
visualize_classifier(classifier, X_test, y_test, 'Тестовий набір даних')

# Обчислення показників ефективності класифікатора
class_names = ['Class-0', 'Class-1']
print ("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print (classification_report (y_train, classifier.predict(X_train), target_names=class_names) )
print ("#"*40 + "\n")

print ("#"*40)
print ("\nClassifier performance on test dataset\n")
print (classification_report(y_test, y_test_pred, target_names=class_names) )
print ("#"*40 + "\n")

plt.show()