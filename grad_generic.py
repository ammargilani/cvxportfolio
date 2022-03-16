import torch
from torch.autograd import grad
from generic import warn


@torch.no_grad()
def set_grad(tensor, new_grad):
    tensor.grad = torch.autograd.Variable(new_grad.clone().detach())


def _check_tensor_shapes(input_shape):
    if len(input_shape) == 0:
        raise Exception('Error! input is a scalar!')


def grad_ud(output, input, create_graph=False, retain_graph=False, allow_unused=False):
    try:
        return grad(output, input, create_graph=create_graph, retain_graph=retain_graph,
                    allow_unused=allow_unused)[0]
    except Exception as e:
        if 'allow_unused=True' in str(e):
            warn("'allow_unused=True' occurred.")
            return torch.zeros(input.shape)
        else:
            raise e


def jacobian_ud(output, input, create_graph=False, retain_graph=False, allow_unused=False):
    output_squeezed = output.squeeze()
    _check_tensor_shapes(input.squeeze().shape)
    output_dim = len(output_squeezed.shape)
    if output_dim == 0:
        return grad_ud(output_squeezed, input, create_graph=create_graph, retain_graph=retain_graph,
                       allow_unused=allow_unused)
    elif output_dim == 1:
        return _jacobian_ud_1d(output_squeezed, input, create_graph=create_graph,
                               retain_graph=retain_graph,
                               allow_unused=allow_unused)
    elif output_dim == 2:
        return _jacobian_ud_2d(output_squeezed, input, create_graph=create_graph,
                               retain_graph=retain_graph,
                               allow_unused=allow_unused)
    else:
        raise Exception('Output dimensionality not understood!')


def _jacobian_ud_1d(output, input, allow_unused, create_graph, retain_graph):
    jacobian = torch.zeros(output.shape + input.squeeze().shape)
    length = jacobian.shape[0]
    for i in range(length - 1):
        jacobian[i] = grad_ud(output[i], input, create_graph=create_graph, retain_graph=True,
                              allow_unused=allow_unused).squeeze()
    jacobian[length - 1] = grad_ud(output[length - 1], input, create_graph=create_graph,
                                   retain_graph=retain_graph,
                                   allow_unused=allow_unused).squeeze()
    return jacobian


def _jacobian_ud_2d(output, input, allow_unused, create_graph, retain_graph):
    jacobian = torch.zeros(output.shape + input.squeeze().shape)
    length = jacobian.shape[0]
    for i in range(length - 1):
        jacobian[i] = _jacobian_ud_1d(output[i], input, allow_unused=allow_unused, \
                                      create_graph=create_graph, retain_graph=True)
    jacobian[length - 1] = _jacobian_ud_1d(output[length - 1], input, allow_unused, create_graph,
                                           retain_graph)
    return jacobian
