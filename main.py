import torch


def cubic_kernel(t):
    t = torch.abs(t)
    return torch.where(
        t <= 1,
        (1.5 * t ** 3) - (2.5 * t ** 2) + 1,
        torch.where(
            t <= 2,
            (-0.5 * t ** 3) + (2.5 * t ** 2) - (4 * t) + 2,
            torch.zeros_like(t)
        )
    )


def cubicInterpol(data, y, x):
    """
    Perform cubic interpolation for 2D data using 2D floating-point arrays of coordinates (x, y).
    Args:
    - data (torch.Tensor): Input 2D data array.
    - y (torch.Tensor): 2D array of y coordinates (floating-point values).
    - x (torch.Tensor): 2D array of x coordinates (floating-point values).

    Returns:
    - torch.Tensor: Interpolated values at the given (x, y) coordinates.
    """

    # Ensure x and y are tensors with requires_grad if needed
    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=True)
    if not torch.is_tensor(y):
        y = torch.tensor(y, requires_grad=True)

    # Get the integer parts of the coordinates
    x_int = torch.floor(x).long()  # Integer part of x
    y_int = torch.floor(y).long()  # Integer part of y

    # Get the fractional parts (for interpolation weights)
    dx = x - x_int.float()
    dy = y - y_int.float()

    # Precompute the 16 neighboring indices for all points
    neighbors_x = torch.stack([x_int + i for i in range(-1, 3)], dim=-1).clamp(0, data.shape[1] - 1)
    neighbors_y = torch.stack([y_int + i for i in range(-1, 3)], dim=-1).clamp(0, data.shape[0] - 1)

    # Gather the 16 neighboring values for all points
    neighbor_values = torch.stack(
        [data[neighbors_y[..., i], neighbors_x[..., j]] for i in range(4) for j in range(4)],
        dim=-1
    ).reshape(*x.shape, 4, 4)

    # Compute weights for x and y using the cubic kernel
    weights_x = torch.stack([cubic_kernel(dx - i) for i in range(-1, 3)], dim=-1)  # (*batch_size, 4)
    weights_y = torch.stack([cubic_kernel(dy - i) for i in range(-1, 3)], dim=-1)  # (*batch_size, 4)

    # Perform the weighted sum explicitly instead of using einsum
    # First, apply weights along the x-dimension (columns)
    weighted_x_sum = torch.zeros_like(dx)
    for i in range(4):
        for j in range(4):
            weighted_x_sum += neighbor_values[..., i, j] * weights_x[..., i]

    # Now apply weights along the y-dimension (rows)
    interpolated_values = torch.zeros_like(dx)
    for i in range(4):
        interpolated_values += weighted_x_sum * weights_y[..., i]

    return interpolated_values


# Example usage
data = torch.randn(10, 10, requires_grad=True)  # Example 2D data
x = torch.tensor([[4.2, 1.5], [3.3, 6.7]])  # Example x coordinates (2D)
y = torch.tensor([[3.7, 5.5], [2.1, 4.4]])  # Example y coordinates (2D)

result = cubicInterpol(data, y, x)
#result.backward()  # This will compute gradients for 'data'

print(result)
