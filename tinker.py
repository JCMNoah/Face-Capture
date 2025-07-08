import tkinter as tk
from tkinter import messagebox

def start_capture():
    try:
        top = int(entry_top.get())
        left = int(entry_left.get())
        width = int(entry_width.get())
        height = int(entry_height.get())
        root.capture_region = {'top': top, 'left': left, 'width': width, 'height': height}
        root.destroy()  # Close the GUI window
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid integer values.")

root = tk.Tk()
root.title("Set Game Capture Region")

tk.Label(root, text="Top (Y coordinate):").grid(row=0, column=0, padx=10, pady=5)
entry_top = tk.Entry(root)
entry_top.insert(0, "200")  # default value
entry_top.grid(row=0, column=1)

tk.Label(root, text="Left (X coordinate):").grid(row=1, column=0, padx=10, pady=5)
entry_left = tk.Entry(root)
entry_left.insert(0, "300")
entry_left.grid(row=1, column=1)

tk.Label(root, text="Width:").grid(row=2, column=0, padx=10, pady=5)
entry_width = tk.Entry(root)
entry_width.insert(0, "300")
entry_width.grid(row=2, column=1)

tk.Label(root, text="Height:").grid(row=3, column=0, padx=10, pady=5)
entry_height = tk.Entry(root)
entry_height.insert(0, "300")
entry_height.grid(row=3, column=1)

start_btn = tk.Button(root, text="Start Capture", command=start_capture)
start_btn.grid(row=4, column=0, columnspan=2, pady=15)

root.mainloop()

# After GUI closes, capture_region is set
GAME_REGION = getattr(root, 'capture_region', {'top': 200, 'left': 300, 'width': 300, 'height': 300})
print("Using capture region:", GAME_REGION)

# Continue with the rest of your script here, using GAME_REGION
