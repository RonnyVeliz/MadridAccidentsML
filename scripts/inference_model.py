import pandas as pd
import joblib

from models.logistic_model import AlcoholLogisticModel


def main():

    model = AlcoholLogisticModel()

    model.load_model("../models_saved/model.pkl")

    nuevo_conductor = {
        "hora_num": 3,
        "sexo": "Hombre",
        "rango_edad": "25-29",
        "tipo_accidente": "Colision frontal-lateral",
        "estado_meteorológico": "Despejado",
        "cod_distrito": 5
    }
    nuevo_conductor_df = pd.DataFrame([nuevo_conductor])

    nuevo_conductor_encoded = pd.get_dummies(
        nuevo_conductor_df,
        columns=[
            "sexo",
            "rango_edad",
            "tipo_accidente",
            "estado_meteorológico"
        ],
        drop_first=True
    )

    columnas_modelo = joblib.load(
        "../models_saved/columnas.pkl"
    )

    nuevo_conductor_final = nuevo_conductor_encoded.reindex(
        columns=columnas_modelo,
        fill_value=0
    )

    prediction = model.predict(nuevo_conductor_final)

    probability = model.predict_proba(nuevo_conductor_final)

    print("===================================")
    print("   RESULTADO DE LA PREDICCIÓN")
    print("===================================")

    if prediction[0] == 1:
        print("Predicción: POSITIVO")
    else:
        print("Predicción: NEGATIVO")

    print(f"Probabilidad de negativo: {probability[0][0]:.2%}")
    print(f"Probabilidad de positivo: {probability[0][1]:.2%}")


if __name__ == "__main__":
    main()