from scripts.animation import create_animation_from_prediction
from scripts.training import make_and_save_prediction, training_routine, create_model


def main() -> None:
    # ==================================================================
    # Load model, train it with newly created data and simulate movement
    # ==================================================================

    model = training_routine(load_weights=False)

    # Draw NN prediction
    z0 = [3, 4, 0, 0] # Initial conditions
    make_and_save_prediction(model, z0)
    create_animation_from_prediction(f"data/generated_movement/DP_train.gif")


if __name__ == '__main__':
    main()