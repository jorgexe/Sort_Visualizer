import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def bubble_sort(data, draw, interval):
    """
    Perform bubble sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                draw(data, ['green' if x == j or x == j+1 else 'blue' for x in range(len(data))])
                plt.pause(interval)
    draw(data, ['green' for _ in range(len(data))])

def selection_sort(data, draw, interval):
    """
    Perform selection sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    n = len(data)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if data[j] < data[min_index]:
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]
        draw(data, ['green' if x == i or x == min_index else 'blue' for x in range(len(data))])
        plt.pause(interval)
    draw(data, ['green' for _ in range(len(data))])

def insertion_sort(data, draw, interval):
    """
    Perform insertion sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    n = len(data)
    for i in range(1, n):
        key = data[i]
        j = i - 1
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
        draw(data, ['green' if x == i or x == j+1 else 'blue' for x in range(len(data))])
        plt.pause(interval)
    draw(data, ['green' for _ in range(len(data))])

def merge_sort(data, draw, interval):
    """
    Perform merge sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    merge_sort_recursive(data, 0, len(data) - 1, draw, interval)

def merge_sort_recursive(data, left, right, draw, interval):
    """
    Recursively perform merge sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - left: int
        The index of the left boundary of the subarray to be sorted.
    - right: int
        The index of the right boundary of the subarray to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.
    """
    if left < right:
        middle = (left + right) // 2
        merge_sort_recursive(data, left, middle, draw, interval)
        merge_sort_recursive(data, middle + 1, right, draw, interval)
        merge(data, left, middle, right, draw, interval)

def merge(data, left, middle, right, draw, interval):
    """
    Merge two sorted halves of the array and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - left: int
        The index of the left boundary of the subarray.
    - middle: int
        The index of the middle element in the subarray.
    - right: int
        The index of the right boundary of the subarray.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.
    """
    left_part = data[left:middle + 1].copy()
    right_part = data[middle + 1:right + 1].copy()

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            data[k] = left_part[i]
            i += 1
        else:
            data[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        data[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        data[k] = right_part[j]
        j += 1
        k += 1

    draw(data, ['green' if left <= x <= right else 'blue' for x in range(len(data))])
    plt.pause(interval)

def quick_sort(data, draw, interval):
    """
    Perform quicksort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    quick_sort_recursive(data, 0, len(data) - 1, draw, interval)

def quick_sort_recursive(data, low, high, draw, interval):
    """
    Recursively perform quicksort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - low: int
        The index of the lowest element in the subarray to be sorted.
    - high: int
        The index of the highest element in the subarray to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.
    """
    if low < high:
        pivot_index = partition(data, low, high, draw, interval)

        quick_sort_recursive(data, low, pivot_index - 1, draw, interval)
        quick_sort_recursive(data, pivot_index + 1, high, draw, interval)

def partition(data, low, high, draw, interval):
    """
    Partition the array and visualize the process in the quicksort algorithm.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - low: int
        The index of the lowest element in the subarray.
    - high: int
        The index of the highest element in the subarray.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.

    Returns:
    - int
        The index of the pivot element after partitioning.
    """
    pivot = data[high]
    i = low - 1

    for j in range(low, high):
        if data[j] < pivot:
            i += 1
            data[i], data[j] = data[j], data[i]
            draw(data, ['green' if x == i or x == j else 'blue' for x in range(len(data))])
            plt.pause(interval)

    data[i + 1], data[high] = data[high], data[i + 1]
    draw(data, ['green' if x == i + 1 or x == high else 'blue' for x in range(len(data))])
    plt.pause(interval)

    return i + 1

def heap_sort(data, draw, interval):
    """
    Perform heapsort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    build_max_heap(data, draw, interval)
    n = len(data)

    for i in range(n - 1, 0, -1):
        data[i], data[0] = data[0], data[i]
        draw(data, ['green' if x == i or x == 0 else 'blue' for x in range(len(data))])
        plt.pause(interval)
        max_heapify(data, 0, i, draw, interval)

def build_max_heap(data, draw, interval):
    """
    Build a max heap from the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.
    """
    n = len(data)

    for i in range(n // 2 - 1, -1, -1):
        max_heapify(data, i, n, draw, interval)

def max_heapify(data, i, n, draw, interval):
    """
    Maintain the max heap property and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - i: int
        The index of the current root node in the heap.
    - n: int
        The size of the heap.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.
    - interval: float
        The time interval between visualization steps.
    """
    largest = i
    left_child = 2 * i + 1
    right_child = 2 * i + 2

    if left_child < n and data[left_child] > data[largest]:
        largest = left_child

    if right_child < n and data[right_child] > data[largest]:
        largest = right_child

    if largest != i:
        data[i], data[largest] = data[largest], data[i]
        draw(data, ['green' if x == i or x == largest else 'blue' for x in range(len(data))])
        plt.pause(interval)
        max_heapify(data, largest, n, draw, interval)

def counting_sort(data, draw, interval):
    """
    Perform counting sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    max_value = max(data)
    min_value = min(data)
    range_of_elements = max_value - min_value + 1

    count = [0] * range_of_elements
    output = [0] * len(data)

    for i in range(len(data)):
        count[data[i] - min_value] += 1

    for i in range(1, len(count)):
        count[i] += count[i - 1]

    i = len(data) - 1
    while i >= 0:
        output[count[data[i] - min_value] - 1] = data[i]
        count[data[i] - min_value] -= 1
        i -= 1

    for i in range(len(data)):
        data[i] = output[i]
        draw(data, ['green' if x == i else 'blue' for x in range(len(data))])
        plt.pause(interval)

def shell_sort(data, draw, interval):
    """
    Perform shell sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process. It should take two parameters:
        - data: List[int]
            The current state of the data.
        - colors: List[str]
            The list of colors corresponding to each element in the data.

    - interval: float
        The time interval between visualization steps.
    """
    n = len(data)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = data[i]
            j = i

            while j >= gap and data[j - gap] > temp:
                data[j] = data[j - gap]
                draw(data, ['green' if x == j or x == j - gap else 'blue' for x in range(len(data))])
                plt.pause(interval)
                j -= gap

            data[j] = temp
            draw(data, ['green' if x == j else 'blue' for x in range(len(data))])
            plt.pause(interval)

        gap //= 2

def radix_sort(data, draw, interval):
    """
    Perform radix sort on the input data and visualize the process.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process.
    - interval: float
        The time interval between visualization steps.
    """
    # Find the maximum number to know the number of digits
    max_num = max(data)

    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_radix(data, draw, interval, exp)
        exp *= 10


def counting_sort_radix(data, draw, interval, exp):
    """
    Perform counting sort on the input data based on the current digit.

    Parameters:
    - data: List[int]
        The list of integers to be sorted.
    - draw: function
        A function to visualize the sorting process.
    - interval: float
        The time interval between visualization steps.
    - exp: int
        The current digit place value.
    """
    n = len(data)
    output = [0] * n
    count = [0] * 10

    # Store count of occurrences in count[]
    for i in range(n):
        index = data[i] // exp
        count[index % 10] += 1

    # Change count[i] so that count[i] now contains actual
    # position of this digit in output[]
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = data[i] // exp
        output[count[index % 10] - 1] = data[i]
        count[index % 10] -= 1
        i -= 1

    # Copy the output array to data[], so that data[] now
    # contains sorted numbers according to the current digit
    for i in range(n):
        data[i] = output[i]
        draw(data, ['green' if x == i else 'blue' for x in range(len(data))])
        plt.pause(interval)

def update(frame, bars, colors):
    """
    Update the colors of the bars in the animation.

    Parameters:
    - frame: int
        The frame number in the animation.
    - bars: List[matplotlib.container.BarContainer]
        List of bar containers representing the bars in the plot.
    - colors: List[str]
        List of colors corresponding to each element in the data.

    Returns:
    - List[matplotlib.container.BarContainer]
        The updated list of bar containers with new colors.
    """
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    return bars


def draw(data, colors):
    """
    Visualize the current state of the data.

    Parameters:
    - data: List[int]
        The current state of the data.
    - colors: List[str]
        The list of colors corresponding to each element in the data.
    """
    if not hasattr(draw, 'bars'):
        draw.bars = plt.bar(range(len(data)), data, color=colors)
        plt.pause(0.001)
    else:
        for bar, color, height in zip(draw.bars, colors, data):
            bar.set_height(height)
            bar.set_color(color)
        plt.pause(0.001)

def main():
    """
    Main function for the sorting visualization program.

    This function takes user input for choosing a sorting algorithm and the number
    of elements in the array. It then initializes the array with random integers,
    selects the appropriate sorting algorithm, and visualizes the sorting process.

    User Input:
    - Sorting algorithm choice (bubble, selection, insertion, merge, quick, heap,
      counting, shell)
    - Number of elements in the array

    Visualization:
    - Displays the initial state of the array.
    - Animates the sorting process using the chosen algorithm.
    - Displays the final sorted array.

    Returns:
    - None
    """
    algorithm = input("Choose a sorting algorithm (bubble, selection, insertion, merge, quick, heap, counting, shell, radix): ").lower()
    n = int(input("Enter the number of elements in the array: "))
    data = np.random.randint(1, 100, size=n)

    if algorithm == 'bubble':
        sort_func = bubble_sort
    elif algorithm == 'selection':
        sort_func = selection_sort
    elif algorithm == 'insertion':
        sort_func = insertion_sort
    elif algorithm == 'merge':
        sort_func = merge_sort
    elif algorithm == 'quick':
        sort_func = quick_sort
    elif algorithm == 'heap':
        sort_func = heap_sort
    elif algorithm == 'counting':
        sort_func = counting_sort
    elif algorithm == 'shell':
        sort_func = shell_sort
    elif algorithm == 'radix':
        sort_func = radix_sort
    else:
        print("Invalid algorithm choice. Exiting.")
        return

    plt.figure(figsize=(10, 6))
    plt.title(f"{algorithm.capitalize()} Sort Visualization")

    sort_func(data.copy(), draw, 0.01)

    plt.show()

if __name__ == "__main__":
    main()

