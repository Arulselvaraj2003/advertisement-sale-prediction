# Advertisement Sale Prediction

A machine learning project that predicts whether a customer is likely
to purchase a product based on their age and estimated salary.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Logistic Regression
- FastAPI
- Docker

## Project Structure

```text
advertisement-sale-prediction/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── app.py
│   └── predict.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model

The project uses Logistic Regression to predict customer purchasing
behavior using age and estimated salary.

The model was evaluated using accuracy, a confusion matrix,
and a classification report.

## Model Performance

- Test accuracy: 86.25%

## Run with Docker

Build the image:

```bash
docker build -t advertisement-sale-prediction:v1 .
```

Start the container:

```bash
docker run -d --name advertisement-api -p 8000:8000 advertisement-sale-prediction:v1
```

Open the API documentation:

http://localhost:8000/docs

## API Endpoint

`POST /predict`

Example request:

```json
{
  "age": 50,
  "estimated_salary": 120000
}
```

The API returns a purchase prediction and associated probabilities.

## Disclaimer

This project is for educational purposes. Predictions are not guarantees
of actual customer purchasing behavior.