import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curves(file_path):
#visualizing loss curves
    data = pd.read_csv(file_path)

    print(f'The best loss of {data["loss"].min()} achieved at Epoch: {data.loc[data["loss"].idxmin(), "epoch"]}')


    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(data['epoch'], data['loss'], label='Training Loss', color='blue')

    # Plot validation loss
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', color='red')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.grid()
    plt.show()