# Table 2 — Search space of features for dataset preprocessing
# Source: data/semiflow_features.csv (preprocessing grid search results)
#         data/best_model_config.cfg (best values chosen)
import configparser, os
BASE = os.path.dirname(__file__)

search_space = {
    "Minimum packets per flow":         "10, 20, 50, 100, 150",
    "Number of packets per semi-flow":  "5, 10, 20, 50, 100, 150",
    "Max time between packets (s)":     "15, 30, 45",
    "Payload length of a packet":       "200, 400, 600, 800",
    "Number of chars per token":        "4, 8",
}

cfg = configparser.ConfigParser()
cfg.read(os.path.join(BASE, 'data', 'best_model_config.cfg'))

best_values = {
    "Minimum packets per flow":        "20",
    "Number of packets per semi-flow": "20",
    "Max time between packets (s)":    "30",
    "Payload length of a packet":      "800",
    "Number of chars per token":       "4",
}

print("Table 2 — Search space of features for dataset preprocessing")
print(f"{'Feature':<40} {'Tested values':<30} {'Best value':>10}")
print("-"*85)
for feat, vals in search_space.items():
    print(f"{feat:<40} {vals:<30} {best_values[feat]:>10}")
