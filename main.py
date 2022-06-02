from scripts.animation import create_animation_from_prediction
from scripts.training import make_and_save_prediction, train_model


def main():
    """
    Load model, train it with newly created data and simulate movement.
    
    """
    model = train_model(load_weights=False, create_data=True)

    # Draw NN prediction
    z0 = [3, 4, 0, 0] # Initial conditions
    make_and_save_prediction(model, z0)
    create_animation_from_prediction(filename=f"data/generated_movement/DP_train.gif")


if __name__ == '__main__':
    main()