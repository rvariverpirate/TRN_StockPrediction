import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class VizHelp():
    # Additional Usefull Display Methods
    def plotPredictions(self, predictions, targets, decoder_steps, epochs, file_name, show=True):
        stock_name = file_name.split('.')[0].upper()
        if '/' in stock_name:
            stock_name = stock_name.split('/')[-1]
        plt.plot(targets, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.xlabel('Samples')
        plt.ylabel('Scaled Price')
        plt.title(f'Price Prediction {stock_name}: D={decoder_steps}, Epoch={epochs}', fontsize=14)
        print(f'Saving image to: plots/Test_{stock_name}D{decoder_steps}_E{epochs}.png')
        plt.savefig(f'plots/Test_{stock_name}_D{decoder_steps}_E{epochs}.png')
        if show:
            plt.show()
        else:
            plt.close()

    def plotMSE(self, errors, decoder_steps, num_samples=1, name='', show=True):
        epoch_counts = [i/num_samples for i in range(len(errors))]
        plt.plot(epoch_counts, errors)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title(f'MSE {name}: D={decoder_steps}', fontsize=14)
        print(f'Saving image to: plots/{name}D{decoder_steps}_E{epoch_counts[-1]}.png')
        plt.savefig(f'plots/{name}_D{decoder_steps}_E{epoch_counts[-1]}.png')
        if show:
            plt.show()
        else:
            plt.close()