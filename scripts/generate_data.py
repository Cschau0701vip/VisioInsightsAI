import json
import numpy as np

data_size = 1000
data = [
    {
        "year": int(np.random.randint(2000, 2023)),
        "mileage": int(np.random.randint(10000, 200000)),
        "condition": np.random.choice(["Excellent", "Very Good", "Good", "Fair", "Poor"]),
        "resale_value": float(np.random.normal(20000, 5000))
    }
    for _ in range(data_size)
]

with open("data/my_data.json", "w") as f:
    json.dump(data, f, indent=4)