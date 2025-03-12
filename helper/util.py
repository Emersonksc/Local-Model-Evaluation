def split_into_columns(my_list, X, Y):
    """
    Splits a flat list (of X*Y elements) into Y lists,
    where the j-th list contains:
        [my_list[j], my_list[j+Y], my_list[j+2Y], ..., my_list[j+(X-1)*Y]]
    
    Parameters:
        my_list (list): A list with X*Y elements.
        X (int): Number of rows.
        Y (int): Number of columns.

    Returns:
        list of lists: Y lists corresponding to the described pattern.
    """
    return [my_list[i::Y] for i in range(Y)]


if __name__ == "__main__":
    X = 4  # Number of rows
    Y = 3  # Number of columns
    # Create a list of numbers from 1 to X*Y (i.e. 1 to 12)
    my_list = list(range(1, X * Y + 1))
    columns = split_into_columns(my_list, X, Y)
    
    for i, col in enumerate(columns, start=1):
        print(f"List {i}: {col}")