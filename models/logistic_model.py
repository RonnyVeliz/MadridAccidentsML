from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib



class AlcoholLogisticModel:
    """
    Clase que encapsula un modelo de Regresión Logística
    para predecir la variable ai_alchool.
    """

    def __init__(self, random_state=42):
        """
        Inicializa el modelo de Regresión Logística.
        """

        self.random_state = random_state

        self.model = LogisticRegression(
            random_state=self.random_state,
            class_weight="balanced",
            max_iter=1000
        )

    def fit(self, X_train, y_train):
        """
        Entrena el modelo con los datos de entrenamiento.
        """

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Realiza predicciones utilizando el modelo entrenado.
        """

        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Devuelve la probabilidad de cada clase.
        """
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el rendimiento del modelo.
        """

        predictions = self.predict(X_test)

        print(classification_report(y_test, predictions))

        print(confusion_matrix(y_test, predictions))

    def save_model(self, path):
        """
        Guarda el modelo entrenado en un archivo.
        """

        joblib.dump(self.model, path)



    def load_model(self, path):
        """
        Carga un modelo previamente entrenado.
        """

        self.model = joblib.load(path)


