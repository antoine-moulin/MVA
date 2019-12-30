import numpy as np
import matplotlib.pyplot as plt
from undirected_chain import UndirectedChain
from typing import List

plt.style.use('ggplot')


class Ising(UndirectedChain):
    """
    Class representing the graphical model of the Ising model. The joint probability of the nodes
    :math:`x_1, \dots, x_n` is given by:

    .. math::
        p(x) = \\frac{1}{Z(\\alpha, \\beta)} \\exp\\left( \\alpha \sum_i x_i + \\beta \sum_{i \sim j} \\mathbb{I}_{x_i=x_j} \\right)

    This class inherits from the class UndirectedChain, the explanations are given in the report. Corresponds to the
    question 2 of the exercise 2 of the homework.

    Attributes
    ----------
    height: int
            Height of the 2D-grid.

    width: int
           Width of the 2D-grid.

    alpha: float
           Coefficient for the node potentials.

    beta: float
          Coefficient for the edge potentials.

    See also UndirectedChain documentation.
    """

    def __init__(self, height: int, width: int, alpha: float, beta: float):
        """
        Initialization of the Ising model. Instead of representing the nodes as a 2D-grid, rows are merged so it can be
        represented as an undirected chain.

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
        self.alpha: float = alpha  # for node potentials
        self.beta: float = beta  # for edge potentials

        indices1 = np.ndindex(*((2,) * width))  # all possible binary arrays of shape width
        indices2 = np.ndindex(*((2,) * width))  # all possible binary arrays of shape width

        # compute the potentials
        log_psis1 = np.zeros((2,) * width)  # node potentials for the undirected chain
        log_psis2 = np.zeros(2 * log_psis1.shape)  # edge potentials for the undirected chain
        for idx1 in indices1:
            log_psis1[idx1] = self.alpha * sum(idx1) + self.beta * sum([idx1[j] == idx1[j+1] for j in range(width-1)])
            for idx2 in indices2:
                log_psis2[idx1, idx2] = self.beta * sum(np.array(idx1) == np.array(idx2))

        # repeat the potentials for all the nodes and edges and then convert to a list
        log_psis1: List[np.ndarray] = list(np.tile(log_psis1.ravel(),
                                                   [height, 1]))  # height nodes
        log_psis2: List[np.ndarray] = list(np.tile(log_psis2.reshape(2**width, 2**width),
                                                   (height-1, 1, 1)))  # height-1 edges

        # create the undirected chain
        super(Ising, self).__init__(log_psis1, log_psis2)


if __name__ == '__main__':
    # parameters
    height = 100
    width = 10
    alpha = 0

    n_values = 21
    beta_values = np.linspace(-2, 2, n_values)
    log_partition_values = np.zeros(beta_values.shape)

    # iterate for different values
    for k in range(len(beta_values)):
        beta = beta_values[k]
        ising = Ising(height, width, alpha, beta)
        ising.compute_marginal_distribution(1, False)
        log_partition_values[k] = ising.partition_function

    # plot the results
    plt.figure()
    plt.title('Log-partition function Z as a function of $\\beta$')
    plt.xlabel('$\\beta$')
    plt.ylabel('$\\log \\, Z \\left( 0, \\beta \\right)$')
    plt.plot(beta_values, log_partition_values, label='$\\log \\, Z$')
    plt.legend(loc='best')
    plt.show()
