import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from typing import List


class UndirectedChain:
    """
    Class representing the graphical model of an undirected chain. Denoting by :math:`\phi` the potentials, the joint
    probability of the nodes :math:`x_1, \dots, x_n` is given by:

    .. math::
        p(x) = \\frac1Z \prod_{i=1}^n \phi_i(x_i) \prod_{i=1}^{n-1} \phi_{i, i+1}(x_i, x_{i+1})

    Everything is represented on the log-scale.
    Corresponds to the question 1 of the exercise 2 of the homework.

    Attributes
    ----------
    n_nodes: int
             Number of nodes in the undirected chain.

    node_potentials: list of ndarrays
                     The element i is a 1D-array corresponding to the different values of the function :math:`\phi_i`.

    edge_potentials: list of ndarrays
                     The element i is a 2D-array corresponding to the different values of the function
                     :math:`\phi_{i, i+1}`.

    forward_messages: list of ndarrays
                      The element i is a 1D-array corresponding to the message :math:`\mu_{i \\rightarrow i+1}`. See the
                      report for the formula.

    backward_messages: list of ndarrays
                       The element i is a 1D-array corresponding to the message :math:`\mu_{i+1 \\rightarrow i}`. See
                       the report for the formula.

    partition_function: float
                        Corresponds to the constant :math:`Z` in the joint distribution.

    marginals: list of ndarrays
               The element i is a 1D-array corresponding to the marginal distribution of the node i. See the report for
               the formula.
    """

    def __init__(self, log_psis1: List[np.ndarray], log_psis2: List[np.ndarray]):
        """
        Initialization of the undirected chain.

        Parameters
        ----------
        log_psis1: list of arrays, len self.n_nodes
                   The 1D-array indexed by i corresponds to the potential (function) of the node i. The arrays can have
                   different sizes (that is why a list is used).

        log_psis2: list of arrays, len self.n_nodes-1
                   The 2D-array indexed by i corresponds to the potential (function) of the edge (i, i+1). The arrays
                   can have different sizes (that is why a list is used).
        """

        # number of nodes in the chain
        self.n_nodes: int = len(log_psis1)

        # lists of the potentials of the nodes and of the edges, respectively
        self.node_potentials: List[np.ndarray] = log_psis1  # element i is a 1D-array associated to the node i
        self.edge_potentials: List[np.ndarray] = log_psis2  # element i is a 2D-array associated to the edge (i, i+1)

        # to make sure the messages are computed when marginalizing
        self._forward_computed: bool = False
        self._backward_computed: bool = False

        # a message's shape is the shape of the node receiving the information
        self.forward_messages: List[np.ndarray] = [
            np.zeros(self.node_potentials[k+1].shape) for k in range(self.n_nodes-1)
        ]
        self.backward_messages: List[np.ndarray] = [
            np.zeros(self.node_potentials[k].shape) for k in range(self.n_nodes-1)
        ]

        # initialization of the marginal distributions and the normalization constant
        self.partition_function: float = 0
        self.marginals: List[np.ndarray] = [
            np.zeros(self.node_potentials[k].shape) for k in range(self.n_nodes)
        ]

    def compute_forward_messages(self, normalize: bool = False):
        """
        Compute the forward messages, starting from the first node and propagating information in the chain to the last
        node.

        Parameters
        ----------
        normalize: bool
                   Boolean to indicate if the messages are normalized or not.

        Returns
        -------
        forward_messages: list, len n_nodes-1
                          For :math:`i = 0, \\dots, n-2`, self.forward_messages[i] is the array
                          :math:`\\mu_{i \\rightarrow i+1}`, with shape self.edge_potentials[i].shape[1] (or
                          self.node_potentials[i+1].shape) that corresponds to the number of values the node
                          :math:`i+1` can take.
        """

        for i in range(self.n_nodes-1):
            node_p = self.node_potentials[i]
            edge_p = self.edge_potentials[i]

            if i != 0:  # if i is not the left-hand extremity, there is a forward message to propagate
                for j in range(edge_p.shape[1]):
                    self.forward_messages[i][j] = logsumexp(node_p + edge_p[:, j] + self.forward_messages[i-1])

            else:  # first forward message
                for j in range(edge_p.shape[1]):
                    self.forward_messages[i][j] = logsumexp(node_p + edge_p[:, j])

        if normalize:
            for i in range(self.n_nodes-1):
                self.forward_messages[i] -= logsumexp(self.forward_messages[i])

        self._forward_computed = True
        return self.forward_messages

    def compute_backward_messages(self, normalize: bool = False):
        """
        Compute the backward messages, starting from the last node and propagating information in the chain to the first
        node.

        Parameters
        ----------
        normalize: bool
                   Boolean to indicate if the backward messages are normalized or not.

        Returns
        -------
        backward_messages: list, len n_nodes-1
                           For :math:`i = 0, \\dots, n-2`, ``self.backward_messages[i]`` is the array
                           :math:`\mu_{i+1 \\rightarrow i}`, with shape ``self.edge_potentials[i].shape[0]`` (or
                           ``self.node_potentials[i].shape``) that corresponds to the number of values the node
                           :math:`i` can take.
        """

        for i in range(self.n_nodes-1, 0, -1):
            node_p = self.node_potentials[i]
            edge_p = self.edge_potentials[i-1]

            if i != self.n_nodes-1:  # if i is not the right-hand extremity, there is a backward message to propagate
                for j in range(edge_p.shape[0]):
                    self.backward_messages[i-1][j] = logsumexp(node_p + edge_p[j, :] + self.backward_messages[i])

            else:  # first backward message
                for j in range(edge_p.shape[0]):
                    self.backward_messages[i-1][j] = logsumexp(node_p + edge_p[j, :])

        if normalize:
            for i in range(self.n_nodes-1):
                self.backward_messages[i] -= logsumexp(self.backward_messages[i])

        self._backward_computed = True
        return self.backward_messages

    def compute_marginal_distribution(self, j: int, normalize: bool = False):
        """
        Compute a marginal distribution using forward and backward messages.

        Parameters
        ----------
        j: int
           Index of the node whose marginal distribution is to be computed.

        normalize: bool
                   Boolean to indicate if the messages are normalized or not.

        Returns
        -------
        marginal: ndarray, shape (self.node_potentials[j].shape,)
                  Marginal distribution of the node j.
        """

        # make sure the messages are all computed
        if not self._forward_computed:
            self.compute_forward_messages(normalize)
        if not self._backward_computed:
            self.compute_backward_messages(normalize)

        self.marginals[j] = self.node_potentials[j]

        # if j is not the left-hand extremity, it receives a forward message
        if j != 0:
            self.marginals[j] += self.forward_messages[j-1]

        # if j is not the right-hand extremity, it receives a backward message
        if j != self.n_nodes-1:
            self.marginals[j] += self.backward_messages[j]

        # the partition function does not depend on j, computing it once is enough
        if not self.partition_function:
            self.partition_function = logsumexp(self.marginals[j])

        # normalize the marginal distribution
        self.marginals[j] = self.marginals[j] - self.partition_function

        return self.marginals[j]


if __name__ == '__main__':
    # node potentials
    range_gaussian = np.linspace(-3, 3, 30)
    psis1 = [
        np.ones(2) / 2,  # Ber(0.5)
        np.exp(- .5 * range_gaussian**2) / np.sqrt(2 * math.pi),  # N(0, 1) distribution
        np.array([0.2, 0.8]),
    ]

    # edges potentials, independent variables
    psis2 = [
        np.ones((len(psis1[0]), len(psis1[1]))),
        np.ones((len(psis1[1]), len(psis1[2]))),
    ]

    log_node_potentials = [np.log(p1) for p1 in psis1]
    log_edge_potentials = [np.log(p2) for p2 in psis2]

    u_chain = UndirectedChain(log_node_potentials, log_edge_potentials)
    for j in range(u_chain.n_nodes):
        u_chain.compute_marginal_distribution(j)

    print('The undirected chain has {} nodes:'.format(u_chain.n_nodes))
    for i in range(u_chain.n_nodes):
        print('  - Node {} can take {} different values.'.format(i, len(u_chain.node_potentials[i])))

    plt.figure()
    plt.title('Marginal distributions')

    plt.subplot(131)
    plt.title('Bernoulli')
    plt.bar([0, 1], np.exp(u_chain.marginals[0]))

    plt.subplot(132)
    plt.title('Gaussian')
    plt.bar(range_gaussian, np.exp(u_chain.marginals[1]))

    plt.subplot(133)
    plt.title('Bernoulli')
    plt.bar([0, 1], np.exp(u_chain.marginals[2]))

    plt.show()
