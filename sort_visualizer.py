import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
import threading
import time

def bubble_sort(data, draw, interval):

    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                draw(data, ['green' if x == j or x == j+1 else 'blue' for x in range(len(data))])
                time.sleep(interval)
    draw(data, ['green' for _ in range(len(data))])

def selection_sort(data, draw, interval):

    n = len(data)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if data[j] < data[min_index]:
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]
        draw(data, ['green' if x == i or x == min_index else 'blue' for x in range(len(data))])
        time.sleep(interval)
    draw(data, ['green' for _ in range(len(data))])

def insertion_sort(data, draw, interval):

    n = len(data)
    for i in range(1, n):
        key = data[i]
        j = i - 1
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
        draw(data, ['green' if x == i or x == j+1 else 'blue' for x in range(len(data))])
        time.sleep(interval)
    draw(data, ['green' for _ in range(len(data))])

def merge_sort(data, draw, interval):

    merge_sort_recursive(data, 0, len(data) - 1, draw, interval)

def merge_sort_recursive(data, left, right, draw, interval):

    if left < right:
        middle = (left + right) // 2
        merge_sort_recursive(data, left, middle, draw, interval)
        merge_sort_recursive(data, middle + 1, right, draw, interval)
        merge(data, left, middle, right, draw, interval)

def merge(data, left, middle, right, draw, interval):

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
    time.sleep(interval)

def quick_sort(data, draw, interval):

    quick_sort_recursive(data, 0, len(data) - 1, draw, interval)

def quick_sort_recursive(data, low, high, draw, interval):

    if low < high:
        pivot_index = partition(data, low, high, draw, interval)

        quick_sort_recursive(data, low, pivot_index - 1, draw, interval)
        quick_sort_recursive(data, pivot_index + 1, high, draw, interval)

def partition(data, low, high, draw, interval):

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
    time.sleep(interval)

    return i + 1

def heap_sort(data, draw, interval):

    build_max_heap(data, draw, interval)
    n = len(data)

    for i in range(n - 1, 0, -1):
        data[i], data[0] = data[0], data[i]
        draw(data, ['green' if x == i or x == 0 else 'blue' for x in range(len(data))])
        time.sleep(interval)
        max_heapify(data, 0, i, draw, interval)

def build_max_heap(data, draw, interval):

    n = len(data)

    for i in range(n // 2 - 1, -1, -1):
        max_heapify(data, i, n, draw, interval)

def max_heapify(data, i, n, draw, interval):

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
        time.sleep(interval)
        max_heapify(data, largest, n, draw, interval)

def counting_sort(data, draw, interval):

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
        time.sleep(interval)

def shell_sort(data, draw, interval):

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
            time.sleep(interval)

        gap //= 2

def radix_sort(data, draw, interval):

    # Find the maximum number to know the number of digits
    max_num = max(data)

    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_radix(data, draw, interval, exp)
        exp *= 10


def counting_sort_radix(data, draw, interval, exp):

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
        time.sleep(interval)

def bucket_sort(data, draw, interval):

    # Number of buckets
    num_buckets = 10

    # Create buckets
    buckets = [[] for _ in range(num_buckets)]

    # Place elements into buckets
    for value in data:
        index = int(value * num_buckets / (max(data) + 1))
        buckets[index].append(value)

    # Sort each bucket and update visualization
    for i, bucket in enumerate(buckets):
        buckets[i] = sorted(bucket)
        draw(data, ['green' if x in range(len(data)) else 'blue' for x in range(len(data))])
        plt.pause(interval)

    # Concatenate the sorted buckets
    data.resize(0)
    for bucket in buckets:
        data = np.concatenate((data, bucket))
        draw(data, ['green' if x in range(len(data)) else 'blue' for x in range(len(data))])
        time.sleep(interval)

def update(frame, bars, colors):

    for bar, color in zip(bars, colors):
        bar.set_color(color)
    return bars


def draw(data, colors):

    if not hasattr(draw, 'bars'):
        draw.bars = plt.bar(range(len(data)), data, color=colors)
        plt.pause(0.001)
    else:
        for bar, color, height in zip(draw.bars, colors, data):
            bar.set_height(height)
            bar.set_color(color)
        plt.pause(0.001)


class SortingVisualizerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Sorting Algorithm Visualizer")
        
        self.algorithm_map = {
            'bubble': bubble_sort,
            'selection': selection_sort,
            'insertion': insertion_sort,
            'merge': merge_sort,
            'quick': quick_sort,
            'heap': heap_sort,
            'counting': counting_sort,
            'shell': shell_sort,
            'radix': radix_sort,
            'bucket': bucket_sort
        }
        self.setup_widgets()
        self.data = np.array([])
        self.sorting = False
        
        
    def setup_widgets(self):
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=10)
        
        self.algorithm_var = tk.StringVar()
        self.algorithm_cb = ttk.Combobox(control_frame, textvariable=self.algorithm_var, 
                                       values=list(self.algorithm_map.keys()), width=15)
        self.algorithm_cb.current(0)
        self.algorithm_cb.pack(side=tk.LEFT, padx=5)
        
        self.size_var = tk.IntVar(value=50)
        self.size_entry = ttk.Entry(control_frame, textvariable=self.size_var, width=10)
        self.size_entry.pack(side=tk.LEFT, padx=5)
        
        self.generate_btn = ttk.Button(control_frame, text="Generate", command=self.generate_data)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_sorting)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def generate_data(self):
        n = self.size_var.get()
        self.data = np.random.randint(1, 100, size=n)
        self.update_plot()
        
    def update_plot(self, colors=None):
        self.plot.clear()
        if colors is None:
            colors = ['blue'] * len(self.data)
        self.plot.bar(range(len(self.data)), self.data, color=colors)
        self.canvas.draw()
        
    def draw(self, data, colors):
        if not self.sorting:
            return
        self.data = data.copy()
        self.master.after(0, self.update_plot, colors)
        
    def start_sorting(self):
        if self.sorting or len(self.data) == 0:
            return
            
        algorithm = self.algorithm_var.get()
        if algorithm not in self.algorithm_map:
            return
            
        self.sorting = True
        self.generate_btn['state'] = 'disabled'
        self.start_btn['state'] = 'disabled'
        
        sort_func = self.algorithm_map[algorithm]
        thread = threading.Thread(target=self.run_sort, args=(sort_func,))
        thread.start()
        
    def run_sort(self, sort_func):
        data_copy = self.data.copy()
        try:
            sort_func(data_copy, self.draw, 0.01)
        finally:
            self.master.after(0, self.on_sort_finish)
            
    def on_sort_finish(self):
        self.sorting = False
        self.generate_btn['state'] = 'normal'
        self.start_btn['state'] = 'normal'
        self.update_plot(['green'] * len(self.data))

if __name__ == "__main__":
    root = tk.Tk()
    app = SortingVisualizerGUI(root)
    root.mainloop()

