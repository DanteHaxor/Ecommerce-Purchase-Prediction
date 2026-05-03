from src.preprocess import load_data, prepare_xy, build_preprocessor
from src.train import train
from src.evalaute import evaluate
from src.utils import save_best_model


def main():
    df = load_data("data/raw/shop_smart_ecommerce.csv")

    X, y = prepare_xy(df)

    preprocessor = build_preprocessor(X)

    pipe, grid, X_train, X_test, y_train, y_test = train(X, y, preprocessor)

    evaluate(pipe, X_test, y_test)

    save_best_model(grid)


if __name__ == "__main__":
    main()
