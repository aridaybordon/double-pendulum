from scripts.animation import create_animation_from_prediction
from scripts.model import make_and_save_prediction, training_routine


def main() -> None:
    # ==================================================================
    # Load model, train it with newly created data and simulate movement
    # ==================================================================

    for i in range(50):
        # Create training data and fit model
        model = training_routine()

        # Compute NN prediction for the double pendulum move
        z0 = [0, 0, 3, 4] # Initial conditions
        make_and_save_prediction(model, z0)
        create_animation_from_prediction(f"data/generated_movement/DP_{i}train.gif")


if __name__ == '__main__':
    main()