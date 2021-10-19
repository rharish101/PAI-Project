import os
import typing
import random

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from matplotlib import cm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0

random.seed(0)

class Model:
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    # Number of points to randomly sample for training
    TRAIN_SIZE = 3000

    # Constants for the GP posterior-based loss-weighted prediction
    PRED_SAMPLES = 100  # For drawing samples from the posterior
    PRED_INTERVAL = 10  # Interval for possible candidates from the posterior

    def __init__(self) -> None:
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        self.model = GaussianProcessRegressor(
            kernel=DotProduct() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-10, 1e3)),
            random_state=self.rng.integers(0, 100),
        )
        self.scaler_y = StandardScaler(with_std=False)

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean, gp_std = self.model.predict(x, return_std=True)
        gp_mean = self.scaler_y.inverse_transform(gp_mean[:, np.newaxis])[:, 0]

        # TODO: Use the GP posterior to form your predictions here

        # Draw samples from the GP's estimation of the true labels
        sample = self.rng.normal(size=(len(x), self.PRED_SAMPLES))
        possible_true = sample * gp_std[:, np.newaxis] + gp_mean[:, np.newaxis]

        # Get possible candidates for the prediction
        possible_pred = np.linspace(
            gp_mean - gp_std, gp_mean + gp_std, num=self.PRED_INTERVAL
        ).T

        # Monte-Carlo approximation of the expected cost per candidate
        all_costs = cost_function(
            # Use broadcasting to avoid for-loops
            np.expand_dims(possible_true, 1),
            np.expand_dims(possible_pred, 2),
            mean=False,
        ).mean(2)

        # Choose the candidate (per data point) with the lowest exptected cost
        best_idxs = all_costs.argmin(1)
        predictions = np.array(
            [possible_pred[i, best_idxs[i]] for i in range(len(x))]
        )

        return predictions, gp_mean, gp_std

    def get_train_data(self, train_x, train_y, sampling_method='uniform', clustering_method='kmeans',
                       intra_cluster_sampling='uniform') -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Sample train data by the given sampling method.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param sampling_method: Sampling method, among ('uniform', 'clustering'). It is set to 'uniform' by default.
        :param clustering_method: The method used for clustering. Possible values: ('kmeans', 'dbscan').
        :param intra_cluster_sampling: How to sample a point in cluster. Possible values: ('uniform', 'medoid').
        """
        if sampling_method == 'uniform':
            indices = self.rng.choice(range(len(train_y)), size=self.TRAIN_SIZE)
            return train_x[indices], train_y[indices]
        elif sampling_method == 'clustering':
            clustering_X = np.column_stack([train_x, train_y])

            if clustering_method == 'kmeans':
                clustering_model = KMeans(n_clusters=self.TRAIN_SIZE, random_state=0)
            elif clustering_method == 'dbscan':
                clustering_model = DBSCAN(eps=0.12, min_samples=1)
            else:
                raise Exception("'{}' is not among supported clustering methods.".format(clustering_method))

            cluster_labels = clustering_model.fit_predict(clustering_X)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"{clustering_method} has been fit on the data and found {n_clusters} clusters.")

            sample_indices = []
            if intra_cluster_sampling == 'uniform':
                for i in range(n_clusters):
                    cluster_sample_index = random.choice(np.argwhere(cluster_labels == i))
                    sample_indices.append(int(cluster_sample_index))
            elif intra_cluster_sampling == 'medoid':
                centroids = []
                for i in range(n_clusters):
                    cluster_indices = [int(ind) for ind in np.argwhere(cluster_labels == i)]
                    centroid = np.mean(clustering_X[cluster_indices], axis=0)
                    centroids.append(centroid)
                centroids = np.array(centroids)
                closest, _ = pairwise_distances_argmin_min(centroids, clustering_X)
                sample_indices = closest
            else:
                print(f"{intra_cluster_sampling} is not among supported intra-cluster sampling methods.")

            return train_x[sample_indices], train_y[sample_indices]
        else:
            raise Exception("'{}' is not among supported sampling methods.".format(sampling_method))

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        train_y = self.scaler_y.fit_transform(train_y[:, np.newaxis])[:, 0]

        X, y = self.get_train_data(train_x, train_y, sampling_method='clustering', clustering_method='kmeans'
                                   , intra_cluster_sampling="medoid")
        self.model.fit(X, y)


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray, mean: bool = True) -> np.ndarray:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    # assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    weighted_cost = cost * weights
    if mean:
        weighted_cost = weighted_cost.mean()
    return weighted_cost


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
