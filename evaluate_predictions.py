import json

from metrics.valid_syntax_rate import evaluate_syntax_rate
from metrics.best_iou import _load_solid_from_code, _normalized_mesh, iou_best

path = "llm_output/mecagents_model_imp.json"
with open(path, "r") as f:
    data = json.load(f)

scores = []

syntax_ds = {str(i): sample["predicted_code"] for i, sample in enumerate(data)}

results = evaluate_syntax_rate(syntax_ds, verbose=False)

ious = []
for sample in data:
    try:
        solid1 = _load_solid_from_code(sample["predicted_code"])
        solid2 = _load_solid_from_code(sample["ground_truth"])
        mesh1 = _normalized_mesh(solid1)
        mesh2 = _normalized_mesh(solid2)
        iou = iou_best(mesh1, mesh2)
        ious.append(iou)
    except Exception as e:
        print(f"Error processing sample: {e}")
        ious.append(0.0)

results["ious"] = ious
results["mean_iou"] = sum(ious) / len(ious)

print(results)

# save results
with open("llm_output/syntax_rate_results_imp.json", "w") as f:
    json.dump(results, f, indent=4)
