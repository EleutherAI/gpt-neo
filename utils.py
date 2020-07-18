from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_n_trainable_vars(graph):
    """
    gets number of trainable vars in a MTF model.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    # Getting total number of trainable vars
    print('\n')
    total_parameters = 0
    for variable in graph.trainable_variables:
      shape = variable.shape.dims
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.size
      total_parameters += variable_parameters
    print("N TRAINABLE VARS:")
    print('{:,}'.format(total_parameters))
    print('\n')


def print_dim_names(graph):
    """

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    all_dim_names = []
    for variable in graph.all_variables:
        names = variable.shape.dimension_names
        all_dim_names.append(names)

    # print all dim names in graph & write to file
    all_dim_names = [item for sublist in all_dim_names for item in sublist] # flatten all dims
    unique_dims = list(set(all_dim_names))
    print("ALL DIM NAMES:")
    with open('all_dim_names.txt', 'w') as f:
        for dim_name in unique_dims:
            f.write("%s\n" % dim_name)
            print(dim_name)
    print('\n')


def get_graph_info(graph):
    """
    wrapper fn that calculates number of trainable vars in an MTF graph & prints all dim_names to file
    TODO: how to get un-trainable dim-names too, batch etc.

    :param graph: Mesh-Tensorflow graph
    :return: None
    """
    get_n_trainable_vars(graph)
    print_dim_names(graph)
