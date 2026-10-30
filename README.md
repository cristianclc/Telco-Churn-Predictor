# Telco Churn Predictor

![CI](https://github.com/cristianclc/telco-churn-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/Deploy-Docker%20%2B%20Railway-2496ED)

Sistema de Machine Learning de punta a punta para predecir el abandono de clientes (*churn*) en una empresa de telecomunicaciones: desde el análisis exploratorio y la comparación de cuatro modelos de *gradient boosting* y *bagging*, hasta su despliegue como API REST en la nube con una interfaz web pública.

**Demo en vivo:** [cristianclc.github.io/telco-churn-predictor](https://cristianclc.github.io/Telco-Churn-Predictor/)

---

## Arquitectura

1. El usuario llena el formulario con los datos del cliente y elige un modelo.
2. La página envía un `POST` a la API alojada en Railway.
3. La API carga el pipeline correspondiente (preprocesamiento + clasificador), calcula la probabilidad y la devuelve.
4. La página muestra el porcentaje de churn, el nivel de riesgo y un gráfico de probabilidad (Chart.js).

## Resultados

Modelos entrenados con `GridSearchCV` (optimizando AUC-ROC, validación cruzada de 5 *folds*) y evaluados en un conjunto de prueba reservado (20 %):

| Modelo | AUC-ROC | Precision | Recall | F1 | Accuracy | Inferencia |
|---|---|---|---|---|---|---|
| **CatBoost** | **0.837** | 0.504 | **0.805** | **0.620** | 0.738 | 0.028 s |
| XGBoost | 0.834 | 0.653 | 0.484 | 0.556 | 0.795 | ~0.020 s |
| Random Forest | 0.833 | 0.516 | 0.738 | 0.607 | 0.746 | 0.071 s |
| LightGBM | 0.833 | **0.662** | 0.487 | 0.561 | **0.797** | ~0.020 s |

- **CatBoost** logra el mejor equilibrio global y el recall más alto: detecta 8 de cada 10 clientes que abandonan. Es la mejor opción cuando el costo de no detectar un abandono es alto.
- **XGBoost y LightGBM** priorizan la precisión y la baja latencia: son útiles cuando las campañas de retención son costosas y se quiere evitar falsos positivos.
- El desbalance de clases se trató con `class_weight` / `scale_pos_weight`.

### Interpretabilidad (LIME)

- **Reducen el riesgo de churn:** contratos de uno o dos años, mayor antigüedad (`tenure`), mayor gasto acumulado (`TotalCharges`) y métodos de pago distintos al cheque electrónico.
- **Aumentan el riesgo de churn:** internet por fibra óptica, cargos mensuales altos y servicio de streaming de películas.

El análisis completo está en los notebooks y en [`docs/conclusiones.md`](docs/conclusiones.md).

## Dataset

[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 clientes con variables demográficas, servicios contratados, tipo de contrato y facturación. Se eliminaron 11 registros con `TotalCharges` vacío.

## Uso de la API

**Endpoint:** `POST https://telcochurn-production.up.railway.app/predict`

Modelos disponibles en `model_name`: `Random Forest`, `XGBoost`, `Catboost`, `LightGBM`.

<details>
<summary>Ejemplo de petición</summary>

```bash
curl -X POST https://telcochurn-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Catboost",
    "data": {
      "SeniorCitizen": 0, "MonthlyCharges": 70.35, "TotalCharges": 1397.475,
      "tenure": 24, "gender": 0, "Partner": 1, "Dependents": 0,
      "PhoneService": 1, "PaperlessBilling": 1,
      "MultipleLines": "Yes", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
      "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
      "Contract": "Month-to-month", "PaymentMethod": "Electronic check"
    }
  }'
```

Respuesta:

```json
{
  "model_used": "Catboost",
  "churn_probability": 0.78,
  "prediction": 1
}
```

Las variables binarias (`gender`, `Partner`, etc.) se envían como `0/1` (`gender`: 0 = femenino, 1 = masculino). Las categóricas usan los valores originales del dataset.
</details>

## Ejecución local


<p align="center">
  <img src="assets/demostracion.gif" width="650" alt="Demostración del formulario de predicción">
</p>


```bash
git clone https://github.com/cristianclc/telco-churn-predictor.git
cd telco-churn-predictor
pip install -r requirements.txt
uvicorn app.api:app --reload
```

**Con Docker:**

```bash
docker build -t telco-churn .
docker run -d -p 8000:8000 telco-churn
```

**Pruebas:**

```bash
pytest tests/
```

## Autores

- **Cristian Linero** · [GitHub](https://github.com/cristianclc)
- **David Marquez**
