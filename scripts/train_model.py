import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from models.logistic_model import AlcoholLogisticModel


def main():
    # Leer dataset limpio generado durante el EDA
    df = pd.read_csv(
        "../data/accidentes_limpio.csv",
        low_memory=False
    )

    # Crear una copia del dataset
    df_clean = df.copy()

    # Eliminar filas sin la variable objetivo
    df_clean = df_clean.dropna(subset=["ai_alchool"])

    # Eliminar registros con edad desconocida
    df_clean = df_clean[
        df_clean["rango_edad"] != "Desconocido"
    ]



    features = [
        "hora_num",
        "sexo",
        "rango_edad",
        "tipo_accidente",
        "estado_meteorológico",
        "cod_distrito"
    ]

    target = "ai_alchool"

    # 4 crear x e y
    X = df_clean[features]
    y = df_clean[target]



    # 5 codificar
    X_encoded = pd.get_dummies(
        X,
        columns=[
            "sexo",
            "rango_edad",
            "tipo_accidente",
            "estado_meteorológico"
        ],
        drop_first=True
    )
    print(X_encoded.columns.to_list())

    # 6. Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42
    )

    # 7. Crear modelo
    model = AlcoholLogisticModel()

    # 8. Entrenar
    model.fit(X_train, y_train)

    model.evaluate(X_test, y_test)

    # 9. Guardar modelo entrenado
    model.save_model("../models_saved/model.pkl")

    joblib.dump(
        X_encoded.columns.tolist(),
        "../models_saved/columnas.pkl"
    )

if __name__ == "__main__":
    main()

