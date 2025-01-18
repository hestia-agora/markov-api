import numpy as np
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from decimal import Decimal, ROUND_DOWN
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()
# Parameters
state_names = [
    "Ingen_undernäring", "Risk_för_undernäring", "Undernäring", "Fallolycka",
    "Trycksår", "Död"
]
n_states = len(state_names)

# Default values for men and women
parameters = {
    "man": {
        "cIU": 0, 
        "cRU": 0, 
        "cU": 130240, 
        "cFO": 273600, 
        "cTS": 550000, 
        "cDeath": 0,

        "oDr": 0.03,  # Discount rate (operational)
        "cDr": 0.03,  # Discount rate (cost)

        "tpDn": 0.08, 
        "tpDRU": 0.15, 
        "tpDU": 0.23, 
        "tpDFO": 0.23, 
        "tpDTS": 0.23,  # Excess probability from trycksår

        "tpIU_RU": 0.0, 
        "tpIU_U": 0.0, 
        "tpIU_FO": 0.0, 
        "tpIU_TS": 0.0,

        "tpRU_IU": 0.05, 
        "tpRU_U": 0.0, 

        "tpU_RU": 0.10, 
        "tpU_FO": 0.00, 
        "tpU_TS": 0.00, 

        "tpFO_RU": 0.03, 
        "tpFO_U": 0.05, 
        "tpFO_TS": 0.01, 

        "tpTS_U": 0.30, 
        "tpTS_FO": 0.15
    },
    "woman": {
        "cIU": 0,
        "cRU": 0,
        "cU": 130240,
        "cFO": 273600,
        "cTS": 550000,
        "cDeath": 0,

        "oDr": 0.03,
        "cDr": 0.03,

        # Risk för död
        "tpDn": 0.08,  # vid ingen undernäring
        "tpDRU": 0.15,  # vid risk för undernäring
        "tpDU": 0.23,  # vid undernäring
        "tpDFO": 0.23,  # vid fallolycka
        "tpDTS": 0.40,  # = excess probability from trycksår

        # Risk för ingen undernäring (IU)
        "tpIU_RU": 0.0,  # till IU från RU - uppdatera om får mer info
        "tpIU_U": 0.0,   # till IU från U - uppdatera om får mer info
        "tpIU_FO": 0.0,  # till IU från FO - uppdatera om får mer info
        "tpIU_TS": 0.0,  # =till IU från TS - uppdatera om får mer info

        # Risk för risk för undernäring (RU)
        "tpRU_IU": 0.05,  # till RU från IU
        "tpRU_U": 0.0,    # till RU från U - uppdatera om får mer info

        # Risk för undernäring (U)
        "tpU_RU": 0.10,  # till U från RU
        "tpU_FO": 0.00,  # till U från FO
        "tpU_TS": 0.00,  # till U från TS

        # Risk för fallolycka (FO)
        "tpFO_RU": 0.04,  # till FO från RU
        "tpFO_U": 0.07,   # till FO från U
        "tpFO_TS": 0.01,  # till FO från TS

        # Risk för trycksår (TS)
        "tpTS_U": 0.30,   # till TS från U
        "tpTS_FO": 0.15   # till TS från FO 
    }
}

def custom_round(value):
    decimal_value = Decimal(value).quantize(Decimal('0.000'))
    third_decimal = int(str(decimal_value)[-1])
    if third_decimal < 5:
        return f"{decimal_value.quantize(Decimal('0.00'), rounding=ROUND_DOWN):.2f}"
    elif third_decimal == 5:
        return str(decimal_value)
    else:
        return f"{decimal_value.quantize(Decimal('0.00')):.2f}"

@app.route('/', methods=['POST'])
def run_model():
    data = request.get_json()
    gender = data.get("gender", "man")
    n_cohort = int(data.get("n_cohort", 1000))
    n_cycles = int(data.get("n_cycles", 10))
    initial_age = int(data.get("Initial_age", 65))

    if gender not in parameters:
        return jsonify({"error": "Invalid gender. Use 'man' or 'woman'."}), 400

    p = parameters[gender] 
    
    # Transition probability matrices
    p_matrix = np.zeros((n_states, n_states, 2))

    # Calculate "Without intervention" matrix using general logic
    p_matrix[:, :, 0] = np.array([
        [1 - p["tpRU_IU"] - p["tpDn"], p["tpRU_IU"], 0, 0, 0, p["tpDn"]],  # Ingen_undernäring
        [0, 1 - p["tpU_RU"] - p["tpDRU"], p["tpU_RU"], p["tpFO_RU"], 0, p["tpDRU"]],  # Risk_för_undernäring
        [0, 0, 1 - p["tpFO_U"] - p["tpTS_U"] - p["tpDU"], p["tpFO_U"], p["tpTS_U"], p["tpDU"]],  # Undernäring
        [0, 0, 0, 1 - p["tpTS_FO"] - p["tpDFO"], p["tpTS_FO"], p["tpDFO"]],  # Fallolycka
        [0, 0, 0, p["tpFO_TS"], 1 - p["tpFO_TS"] - p["tpDTS"], p["tpDTS"]],  # Trycksår
        [0, 0, 0, 0, 0, 1]  # Död
    ])

    effect = data.get("effect", 0.3)

    p_matrix[:, :, 1] = np.array([
        [1 - p["tpRU_IU"] * (1 - effect) - p["tpDn"], p["tpRU_IU"] * (1 - effect), 0, 0, 0, p["tpDn"]],
        [0, 1 - p["tpU_RU"] * (1 - effect) - p["tpDRU"], p["tpU_RU"] * (1 - effect), p["tpFO_RU"], 0, p["tpDRU"]],
        [0, 0, 1 - p["tpFO_U"] - p["tpTS_U"] - p["tpDU"], p["tpFO_U"], p["tpTS_U"], p["tpDU"]],
        [0, 0, 0, 1 - p["tpTS_FO"] - p["tpDFO"], p["tpTS_FO"], p["tpDFO"]],
        [0, 0, 0, p["tpFO_TS"], 1 - p["tpFO_TS"] - p["tpDTS"], p["tpDTS"]],
        [0, 0, 0, 0, 0, 1]
    ])

    # Initialize populations and costs
    population = np.zeros((n_states, n_cycles, 2))
    population[:, 0, :] = np.tile(np.array([n_cohort, 0, 0, 0, 0, 0]), (2, 1)).T

    cycle_costs = np.zeros((2, n_cycles))
    total_costs = np.zeros(2)

    state_costs = np.array([
        [p['cIU'], p['cRU'], p['cU'], p['cFO'], p['cTS'], p['cDeath']],
        [p['cIU'], p['cRU'], p['cU'], p['cFO'], p['cTS'], p['cDeath']]
    ])

    # Run simulation
    for i in range(2):
        for j in range(1, n_cycles):
            population[:, j, i] = np.dot(population[:, j - 1, i], p_matrix[:, :, i])


        discounted_costs = np.dot(state_costs[i, :], population[:, :, i]) / (1 + p['cDr']) ** np.arange(n_cycles)
        cycle_costs[i, :] = discounted_costs
        total_costs[i] = np.sum(discounted_costs)

    # Prepare results
    tp_without_intervention = [[custom_round(value) for value in row] for row in p_matrix[:, :, 0]]
    tp_with_intervention = [[custom_round(value) for value in row] for row in p_matrix[:, :, 1]]

    results = {
        "Transition Probabilities": {
            "utan_insats": tp_without_intervention,
            "med_insats": tp_with_intervention
        },
        "Population Results": {
            "utan_insats": population[:, :, 0].tolist(),
            "med_insats": population[:, :, 1].tolist()
        },
        "Cycle Costs": {
            "utan_insats": cycle_costs[0, :].tolist(),
            "med_insats": cycle_costs[1, :].tolist()
        },
        "Total Costs": {
            "utan_insats": f"{round(total_costs[0]):,}",
            "med_insats": f"{round(total_costs[1]):,}"
        }
    }

    return jsonify(results)


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)
