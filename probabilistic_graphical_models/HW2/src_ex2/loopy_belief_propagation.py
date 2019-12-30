import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class IsingLoopyBeliefPropagation:
    """
    Class representing the graphical model of the Ising model. The joint probability of the nodes
    :math:`x_1, \dots, x_n` is given by:

    .. math::
        p(x) = \\frac{1}{Z(\\alpha, \\beta)} \\exp\\left( \\alpha \sum_i x_i + \\beta \sum_{i \sim j} \\mathbb{I}_{x_i=x_j} \\right)

    This class represents the model as a 2D-grid, unlike the Ising class. Corresponds to the question 3 of the exercise
    2 of the homework.

    Attributes
    ----------
    height: int
            Height of the 2D-grid.

    width: int
           Width of the 2D-grid.

    n_nodes: int
             Number of nodes in the chain.

    alpha: float
           Coefficient for the node potentials.

    beta: float
          Coefficient for the edge potentials.

    adjacency: ndarray, shape (n_nodes, n_nodes)
               Adjacency matrix representing the 2D-grid. adjacency[i, j] = 1 if i and j are neighbors, 0 otherwise.

    node_potentials: ndarray, shape (n_nodes, 2)
                     Element i corresponds to the different values of :math:`\phi_i`.

    edge_potentials: ndarray, shape (n_nodes, n_nodes, 4)
                     If adjacency[i, j] == 1, element (i, j) corresponds to the different values of :math:`\phi_{i, j}`.

    messages: ndarray, shape (n_nodes, n_nodes, 2)
              If adjacency[i, j] == 1, messages[i, j] is the message sent from i to j (can take two different values).

    marginals: ndarray, shape (n_nodes, 2)
               Marginal distributions of each node.

    partition_function: float
                        Corresponds to the normalization constant :math:`Z(\alpha, \beta)` in the joint distribution.
    """

    def __init__(self, height: int, width: int, alpha: float, beta: float):
        """
        Initialization of the Ising model.

        Parameters
        ----------
        height: int
                Height of the 2D-grid.

        width: int
               Width of the 2D-grid.

        alpha: float
               Coefficient for the node potentials.

        beta: float
              Coefficient for the edge potentials.
        """

        # parameters of the Ising model
        self.height: int = height
        self.width: int = width
        self.n_nodes: int = self.height * self.width
        self.alpha: float = alpha
        self.beta: float = beta

        # adjacency matrix of the grid
        self.adjacency: np.ndarray = np.zeros((self.n_nodes, self.n_nodes))
        self._build_grid()

        # potentials
        self.node_potentials: np.ndarray = np.tile([1, np.exp(self.alpha)], (self.n_nodes, 1))
        self.node_potentials = np.log(self.node_potentials)  # log-scale

        self.edge_potentials: np.ndarray = np.zeros((self.n_nodes, self.n_nodes, 4))
        self.edge_potentials[self.adjacency != 0, :] = np.exp(self.beta * np.eye(2).ravel())
        self.edge_potentials[self.adjacency != 0, :] = np.log(self.edge_potentials[self.adjacency != 0, :])  # log-scale

        # beliefs, was here for a test but not used anymore
        self.node_beliefs: np.ndarray = np.zeros((self.n_nodes, 2))
        self.edge_beliefs: np.ndarray = np.zeros((self.n_nodes, self.n_nodes, 4))

        # messages, marginals and partition function, stored on the log-scale
        self.messages: np.ndarray = np.zeros((self.n_nodes, self.n_nodes, 2))
        self.marginals: np.ndarray = np.zeros(self.node_potentials.shape)
        self.partition_function: float = 0

    def _build_grid(self):
        """
        Builds the 2D-grid of the Ising model.
        """

        for k in range(self.n_nodes):
            i, j = k // self.width, k % self.width
            if i-1 >= 0:
                self.adjacency[k, (i - 1) * self.width + j] = 1
            if i+1 <= self.height-1:
                self.adjacency[k, (i + 1) * self.width + j] = 1
            if j-1 >= 0:
                self.adjacency[k, i * self.width + j - 1] = 1
            if j+1 <= self.width-1:
                self.adjacency[k, i * self.width + j + 1] = 1

    def _get_neighbors(self, node: int):
        """
        Retrieve the neighbors of a given node.

        Parameters
        ----------
        node: int
              The node represented by its index.

        Returns
        -------
        neighbors: ndarray, shape (n_nodes,)
                   A binary array. If neighbors[j] == 1, i and j are neighbors, otherwise they are not.
        """

        unit_vector = np.zeros(self.n_nodes, dtype=int)
        unit_vector[node] = 1
        return self.adjacency.dot(unit_vector)

    def _message_initializations(self):
        """
        Initialize the messages.

        Returns
        -------
        messages: ndarray, shape (n_nodes, n_nodes, 2)
                  Initialized messages.
        """

        messages = np.tile(self.adjacency, (2, 1, 1))
        self.messages = messages.transpose(1, 2, 0)

        self.messages[self.messages != 0] = np.log(self.messages[self.messages != 0])  # log-scale

        return self.messages

    def _message_updates(self, normalize: bool = False):
        """
        Update the messages.

        Parameters
        ----------
        normalize: bool
                   Boolean that indicates if the messages must be normalized or not.

        Returns
        -------
        messages: ndarray, shape (n_nodes, n_nodes, 2)
                  Updated messages.
        """

        for i in range(self.n_nodes):  # for each node ...
            node_p = self.node_potentials[i]
            neighbors = np.where(self._get_neighbors(i) != 0)[0]

            for j in neighbors:  # ... send a message to all the neighbors ...
                edge_p = self.edge_potentials[i, j]
                other_neighbors = np.zeros(self.n_nodes, dtype=int)
                other_neighbors[neighbors] = 1
                other_neighbors[j] = 0

                # ... using the formula
                self.messages[i, j, 0] = logsumexp(node_p + edge_p[:2] + self.messages[other_neighbors, i])
                self.messages[i, j, 1] = logsumexp(node_p + edge_p[2:] + self.messages[other_neighbors, i])

                if normalize:
                    self.messages[i, j] -= logsumexp(self.messages[i, j])
            # if normalize:
            #     self.messages[:, i] -= logsumexp(self.messages[:, i], axis=0)
            # if normalize:
            #     self.messages[i, :] -= logsumexp(self.messages[i, :], axis=0)

        return self.messages

    def _compute_beliefs(self):
        """
        Compute the beliefs. NOT USED.
        """

        for i in range(self.n_nodes):
            neighbors_i = self._get_neighbors(i).astype(bool)

            # compute node belief
            self.node_beliefs[i] = logsumexp(self.messages[neighbors_i, i], axis=0)
            self.node_beliefs[i] /= self.node_beliefs[i].sum()

            for j in np.where(neighbors_i != 0)[0]:
                neighbors_j = self._get_neighbors(j).astype(bool)
                msg_to_i = np.sum(self.messages[neighbors_i, i], axis=0) - self.messages[j, i]
                msg_to_i = (msg_to_i.reshape(-1, 1) + msg_to_i.reshape(1, -1)).ravel()
                msg_to_j = np.sum(self.messages[neighbors_j, j], axis=0) - self.messages[i, j]
                msg_to_j = (msg_to_j.reshape(-1, 1) + msg_to_j.reshape(1, -1)).ravel()

                # compute edge belief
                self.edge_beliefs[i, j] = logsumexp(self.edge_potentials[i, j] + msg_to_i + msg_to_j)
                self.edge_beliefs[i, j] /= self.edge_beliefs[i, j].sum()

    def message_passing(self, n_passes: int = 10, normalize: bool = False):
        """
        Run the loopy belief propagation algorithm. Build the grid, initialize the messages and propagate them through
        the graph.

        Parameters
        ----------
        n_passes: int
                  Number of passes for propagation.

        normalize: bool
                   Boolean that indicates if the messages must be normalized or not.
        """

        self._build_grid()
        self._message_initializations()

        for k in range(n_passes):
            self._message_updates(normalize)

        # beliefs not used anymore
        # self._compute_beliefs()

    def compute_marginal_distribution(self, j):
        """
        Compute a marginal distribution using the messages.

        Parameters
        ----------
        j: int
           Index of the node whose marginal distribution is to be computed.

        Returns
        -------
        marginals[j]: ndarray, shape (2,)
                      Marginal distribution of the node j.
        """

        self.marginals[j] = self.node_potentials[j]
        neighbors = np.where(self._get_neighbors(j) != 0)[0]

        for neighbor in neighbors:
            self.marginals[j] += self.messages[neighbor, j]

        # the partition function is different, depending on the number of passes, and the marginal computed
        # because it is an approximation
        self.partition_function = logsumexp(self.marginals[j])

        # normalize the marginal distribution
        self.marginals[j] = self.marginals[j] - self.partition_function

        return self.marginals[j]


if __name__ == '__main__':
    height = 100
    width = 10
    alpha = 0

    n_values = 21
    n_passes = 10  # with 10 passes, as long as the exact computation
    beta_values = np.linspace(-2, 2, n_values)
    log_partition_bethe = np.zeros(beta_values.shape)

    # iterate for different values
    for k in range(n_values):
        beta = beta_values[k]
        ising = IsingLoopyBeliefPropagation(height, width, alpha, beta)
        ising.message_passing(n_passes, False)
        ising.compute_marginal_distribution(45)
        log_partition_bethe[k] = ising.partition_function

    # plot the results
    plt.figure()
    plt.title('Log-partition function $Z_{Bethe}$ as a function of $\\beta$')
    plt.xlabel('$\\beta$')
    plt.ylabel('$\\log \\, Z_{Bethe} \\left( 0, \\beta \\right)$')
    plt.plot(beta_values, log_partition_bethe, label='$\\log \\, Z$')
    plt.legend(loc='best')
    plt.show()
