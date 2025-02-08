import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS 
from decimal import Decimal, ROUND_DOWN
import os

app = Flask(__name__)
CORS(app) 

state_names = [
    "Ingen_undernäring",
    "Risk_för_undernäring",
    "Undernäring",
    "Fallolycka",
    "Trycksår",
    "Död",
]
n_states = len(state_names)


# Function to round values properly
def custom_round(value):
    decimal_value = Decimal(value).quantize(Decimal("0.000"))
    third_decimal = int(str(decimal_value)[-1])
    if third_decimal < 5:
        return f"{decimal_value.quantize(Decimal('0.00'), rounding=ROUND_DOWN):.2f}"
    elif third_decimal == 5:
        return str(decimal_value)
    else:
        return f"{decimal_value.quantize(Decimal('0.00')):.2f}"


@app.route("/", methods=["POST"])
def run_model():
    data = request.get_json()
    print("Received data:", data)  

    # Ensure all required keys exist
    required_general_keys = ["gender", "n_cohort", "n_cycles", "Initial_age", "effect"]
    for key in required_general_keys:
        if key not in data:
            return jsonify({"error": f"Missing key: {key}"}), 400

    # Fetch general parameters
    gender = data["gender"]
    n_cohort = int(data["n_cohort"])
    n_cycles = int(data["n_cycles"])
    initial_age = int(data["Initial_age"])
    effect = float(data["effect"])  # Intervention effect

    # Get user-defined parameters for selected gender
    parameters = data.get("parameters", {})
    costs = data.get("costs", {})

    # Ensure required parameters exist
    required_keys = [
        "tpDn", "tpDRU", "tpDU", "tpDFO", "tpDTS", "tpIU_RU", "tpIU_U", "tpIU_FO",
        "tpIU_TS", "tpRU_IU", "tpRU_U", "tpU_RU", "tpU_FO", "tpU_TS", "tpFO_RU",
        "tpFO_U", "tpFO_TS", "tpTS_U", "tpTS_FO", "cIU", "cRU", "cU", "cFO", "cTS",
        "cDeath", "cDr"
    ]

    for key in required_keys:
        if key not in parameters and key not in costs:
            return jsonify({"error": f"Missing parameter: {key}"}), 400

    # Discount rate (ensuring it has a float value)
    cDr = float(costs.get("cDr", 0.03))  # Default to 3%

    # Transition probability matrices
    p_matrix = np.zeros((n_states, n_states, 2))

    # Assign values from frontend inputs
    p = {**parameters, **costs}

    # Transition probabilities WITHOUT intervention
    p_matrix[:, :, 0] = np.array([
        [1 - p["tpRU_IU"] - p["tpDn"], p["tpRU_IU"], 0, 0, 0, p["tpDn"]],
        [0, 1 - p["tpU_RU"] - p["tpDRU"], p["tpU_RU"], p["tpFO_RU"], 0, p["tpDRU"]],
        [0, 0, 1 - p["tpFO_U"] - p["tpTS_U"] - p["tpDU"], p["tpFO_U"], p["tpTS_U"], p["tpDU"]],
        [0, 0, 0, 1 - p["tpTS_FO"] - p["tpDFO"], p["tpTS_FO"], p["tpDFO"]],
        [0, 0, 0, p["tpFO_TS"], 1 - p["tpFO_TS"] - p["tpDTS"], p["tpDTS"]],
        [0, 0, 0, 0, 0, 1],
    ])

    # Transition probabilities WITH intervention (applying effect)
    p_matrix[:, :, 1] = np.array([
        [1 - p["tpRU_IU"] * (1 - effect) - p["tpDn"], p["tpRU_IU"] * (1 - effect), 0, 0, 0, p["tpDn"]],
        [0, 1 - p["tpU_RU"] * (1 - effect) - p["tpDRU"], p["tpU_RU"] * (1 - effect), p["tpFO_RU"], 0, p["tpDRU"]],
        [0, 0, 1 - p["tpFO_U"] - p["tpTS_U"] - p["tpDU"], p["tpFO_U"], p["tpTS_U"], p["tpDU"]],
        [0, 0, 0, 1 - p["tpTS_FO"] - p["tpDFO"], p["tpTS_FO"], p["tpDFO"]],
        [0, 0, 0, p["tpFO_TS"], 1 - p["tpFO_TS"] - p["tpDTS"], p["tpDTS"]],
        [0, 0, 0, 0, 0, 1],
    ])

    # Initialize population and costs
    population = np.zeros((n_states, n_cycles, 2))
    population[:, 0, :] = np.tile(np.array([n_cohort, 0, 0, 0, 0, 0]), (2, 1)).T

    cycle_costs = np.zeros((2, n_cycles))
    total_costs = np.zeros(2)

    state_costs = np.array([
        [p["cIU"], p["cRU"], p["cU"], p["cFO"], p["cTS"], p["cDeath"]],
        [p["cIU"], p["cRU"], p["cU"], p["cFO"], p["cTS"], p["cDeath"]],
    ])

    # Run simulation
    for i in range(2):  # 0 = without intervention, 1 = with intervention
        for j in range(1, n_cycles):
            population[:, j, i] = np.dot(population[:, j - 1, i], p_matrix[:, :, i])

        # Calculate discounted costs
        discounted_costs = np.dot(state_costs[i, :], population[:, :, i]) / (1 + p["cDr"]) ** np.arange(n_cycles)
        cycle_costs[i, :] = discounted_costs
        total_costs[i] = np.sum(discounted_costs)

    savings = total_costs[0] - total_costs[1]

    # Prepare results
    results = {
        "Transition Probabilities": {
            "utan_insats": [[custom_round(value) for value in row] for row in p_matrix[:, :, 0]],
            "med_insats": [[custom_round(value) for value in row] for row in p_matrix[:, :, 1]],
        },
        "Population Results": {
            "utan_insats": population[:, :, 0].tolist(),
            "med_insats": population[:, :, 1].tolist(),
        },
        "Cycle Costs": {
            "utan_insats": cycle_costs[0, :].tolist(),
            "med_insats": cycle_costs[1, :].tolist(),
        },
        "Total Costs": {
            "utan_insats": f"{round(total_costs[0]):,}",
            "med_insats": f"{round(total_costs[1]):,}",
        },
        "Savings": {"Besparing": f"{round(savings):,} SEK"},
    }

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
