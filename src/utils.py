import pickle
import os


def save_best_model(grid, save_path="models/best_model.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_f1": grid.best_score_,
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print("Best F1:", grid.best_score_)
    print("Best params:", grid.best_params_)
    print("Saved to", save_path)
