from metrics.valid_syntax_rate import evaluate_syntax_rate

import json

path = "llm_output/mecagents_model_imp.json"
with open(path, "r") as f:
    data = json.load(f)

scores = []

syntax_ds = {str(i): sample["predicted_code"] for i, sample in enumerate(data)}

results = evaluate_syntax_rate(syntax_ds, verbose=False)

print(results)

# save results
with open("llm_output/syntax_rate_results_imp.json", "w") as f:
    json.dump(results, f, indent=4)
