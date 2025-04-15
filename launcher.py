#!/usr/bin/env python3
"""
Launcher for Finger Drawing Webcam Application
This script provides a simple GUI to launch different versions of the application.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import importlib.util

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['numpy', 'opencv-python', 'mediapipe']
    missing_packages = []
    
    for package in required_packages:
        # Check if package is installed
        if importlib.util.find_spec(package.replace('-python', '')) is None:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies."""
    try:
        import pip
        for package in missing_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_app(app_file):
    """Run the selected application."""
    try:
        subprocess.Popen([sys.executable, app_file])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch application: {e}")

def run_tests():
    """Run the test utilities."""
    try:
        subprocess.call([sys.executable, "test_utils.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run tests: {e}")

def create_launcher_gui():
    """Create a simple GUI for the launcher."""
    root = tk.Tk()
    root.title("Finger Drawing App Launcher")
    root.geometry("500x400")
    root.resizable(False, False)
    
    # Set background color
    root.configure(bg="#f0f0f0")
    
    # Header
    header_frame = tk.Frame(root, bg="#4a7abc", padx=10, pady=10)
    header_frame.pack(fill="x")
    
    header_label = tk.Label(
        header_frame, 
        text="Finger Drawing Webcam Application", 
        font=("Arial", 16, "bold"),
        fg="white",
        bg="#4a7abc"
    )
    header_label.pack()
    
    # Main content
    content_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
    content_frame.pack(fill="both", expand=True)
    
    description = tk.Label(
        content_frame,
        text="Draw with your finger using your webcam.\nUse your index finger to draw and palm to erase.",
        font=("Arial", 12),
        justify="center",
        bg="#f0f0f0"
    )
    description.pack(pady=10)
    
    # App selection frame
    select_frame = tk.LabelFrame(content_frame, text="Launch Application", padx=10, pady=10, bg="#f0f0f0")
    select_frame.pack(fill="x", pady=10)
    
    # Basic version button
    basic_btn = ttk.Button(
        select_frame, 
        text="Basic Version",
        command=lambda: run_app("app.py")
    )
    basic_btn.pack(fill="x", pady=5)
    
    # Enhanced version button
    enhanced_btn = ttk.Button(
        select_frame, 
        text="Enhanced Version (Color Picker & More Features)",
        command=lambda: run_app("enhanced_app.py")
    )
    enhanced_btn.pack(fill="x", pady=5)
    
    # Test utilities button
    test_btn = ttk.Button(
        content_frame, 
        text="Run Tests (Check Setup)",
        command=run_tests
    )
    test_btn.pack(fill="x", pady=5)
    
    # Footer with instructions
    instructions = tk.Label(
        content_frame,
        text="Controls:\nIndex finger to draw | Open palm to erase\nESC to exit | 'c' to clear canvas",
        font=("Arial", 10),
        justify="center",
        bg="#f0f0f0",
        fg="#666666"
    )
    instructions.pack(side="bottom", pady=10)
    
    return root

def main():
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        if not install_dependencies(missing_packages):
            print("Failed to install dependencies. Please install them manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
        
        print("Dependencies installed successfully.")
    
    # Create and run the GUI
    root = create_launcher_gui()
    root.mainloop()

if __name__ == "__main__":
    main()
