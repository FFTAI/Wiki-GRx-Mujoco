from pynput import keyboard
import torch

# Define the callback functions for key press and release events
def on_press(key):
    global policy
    try:
        if key.char == '.':
            policy = torch.jit.load('../policy/stand_model_jit.pt')
        elif key.char == '/':
            policy = torch.jit.load('../policy/walk_model_jit.pt')
        elif key.char == '0':
            return False  # Stop the listener
    except AttributeError:
        pass