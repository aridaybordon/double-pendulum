from scripts.animation import create_animation_from_prediction
from scripts.model import make_and_save_prediction, training_routine


def main() -> None:
    # ==================================================================
    # Load model, train it with newly created data and simulate movement
    # ==================================================================
    iterations = 1000

    for i in range(iterations):
        print(f"\nIteration {i+1} of {iterations}\n")
        # Create training data and fit model
        model = training_routine(load_weights=i)

        if i%50 == 0:
            # Draw NN prediction
            z0 = [3, 4, 0, 0] # Initial conditions
            make_and_save_prediction(model, z0)
            create_animation_from_prediction(f"data/generated_movement/DP_{i}train.gif")


if __name__ == '__main__':
    main()