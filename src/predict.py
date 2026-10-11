import joblib
import pandas as pd
from pathlib import Path


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ==================================================
# LOAD MODEL AND SCALER
# ==================================================

print("Loading trained model and scaler...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")
print("Scaler loaded successfully!")


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_purchase(age, estimated_salary):

    # Create customer data with the same
    # feature names used during training
    customer = pd.DataFrame(
        [[age, estimated_salary]],
        columns=["Age", "EstimatedSalary"]
    )

    # Scale the customer data
    customer_scaled = scaler.transform(customer)

    # Make prediction
    prediction = model.predict(customer_scaled)

    # Get prediction probabilities
    probability = model.predict_proba(customer_scaled)

    # Convert prediction into readable text
    if prediction[0] == 1:
        result = "Purchased"
    else:
        result = "Not Purchased"

    # Display result
    print()
    print("=" * 45)
    print("       ADVERTISEMENT SALE PREDICTION")
    print("=" * 45)

    print(f"Age                     : {age}")
    print(f"Estimated Salary        : {estimated_salary}")
    print(f"Prediction              : {result}")

    print(
        f"Not Purchased Probability : "
        f"{probability[0][0]:.2%}"
    )

    print(
        f"Purchased Probability     : "
        f"{probability[0][1]:.2%}"
    )

    print("=" * 45)

    return result


# ==================================================
# MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 45)
    print("   ADVERTISEMENT SALE PREDICTION")
    print("=" * 45)

    while True:

        # ------------------------------------------
        # Get customer information
        # ------------------------------------------

        try:

            age = float(input("\nEnter Age: "))

            estimated_salary = float(
                input("Enter Estimated Salary: ")
            )

        except ValueError:

            print(
                "\nInvalid input. "
                "Please enter numbers only."
            )

            continue

        # ------------------------------------------
        # Validate input
        # ------------------------------------------

        if age <= 0:

            print(
                "\nAge must be greater than 0."
            )

            continue

        if estimated_salary < 0:

            print(
                "\nEstimated salary cannot be negative."
            )

            continue

        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        predict_purchase(
            age,
            estimated_salary
        )

        # ------------------------------------------
        # Ask for another prediction
        # ------------------------------------------

        while True:

            choice = input(
                "\nDo you want to predict another "
                "customer? (y/n): "
            ).lower().strip()

            if choice in ["y", "yes"]:

                break

            elif choice in ["n", "no"]:

                print()
                print(
                    "Thank you for using "
                    "Advertisement Sale Prediction!"
                )

                break

            else:

                print(
                    "Please enter 'y' or 'n'."
                )

        # ------------------------------------------
        # Exit if user selected no
        # ------------------------------------------

        if choice in ["n", "no"]:

            break