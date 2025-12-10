import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import subprocess
import threading
import glob

# Constants
OUTPUT_DIR = "final_output"
DATA_DIR = "dataset/original_images"

class UrbanSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏙️ Unsupervised Urban Scene Segmentation")
        self.root.geometry("1200x800")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main Layout
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sidebar
        self.sidebar = ttk.Frame(self.main_container, width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Sidebar Content
        ttk.Label(self.sidebar, text="Controls", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Hyperparameters
        labelframe = ttk.LabelFrame(self.sidebar, text="Hyperparameters")
        labelframe.pack(fill=tk.X, pady=5, padx=5)
        
        # Mode Selection
        self.mode_var = tk.StringVar(value="manual")
        ttk.Label(labelframe, text="Cluster Mode:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(labelframe, text="Auto (Elbow)", variable=self.mode_var, value="auto", command=self.toggle_k_slider).pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(labelframe, text="Manual", variable=self.mode_var, value="manual", command=self.toggle_k_slider).pack(anchor=tk.W, padx=10)
        
        # K Slider
        ttk.Label(labelframe, text="Clusters (K):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.k_var = tk.IntVar(value=5)
        self.k_slider = tk.Scale(labelframe, from_=2, to=15, orient=tk.HORIZONTAL, variable=self.k_var)
        self.k_slider.pack(fill=tk.X, padx=5)
        
        # Patch Size
        ttk.Label(labelframe, text="Patch Size:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.patch_var = tk.IntVar(value=64)
        self.patch_spin = ttk.Spinbox(labelframe, from_=16, to=128, increment=16, textvariable=self.patch_var)
        self.patch_spin.pack(fill=tk.X, padx=5, pady=5)

        # Max Images
        ttk.Label(labelframe, text="Max Images:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.max_img_var = tk.IntVar(value=50)
        self.max_img_spin = ttk.Spinbox(labelframe, from_=10, to=500, increment=10, textvariable=self.max_img_var)
        self.max_img_spin.pack(fill=tk.X, padx=5, pady=5)
        
        self.run_btn = ttk.Button(self.sidebar, text="🚀 Run Analysis", command=self.run_analysis)
        self.run_btn.pack(fill=tk.X, pady=20)
        
        self.status_text = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.sidebar, textvariable=self.status_text, wraplength=200)
        self.status_label.pack(side=tk.BOTTOM, pady=10)
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Tabs
        self.tab_eda = ttk.Frame(self.notebook)
        self.tab_comparison = ttk.Frame(self.notebook)
        self.tab_segmentation = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_eda, text="📊 EDA & Optimization")
        self.notebook.add(self.tab_comparison, text="📝 Model Comparison")
        self.notebook.add(self.tab_segmentation, text="🗺️ Segmentation Results")
        
        # Initialize
        self._last_eda_width = 0
        self._last_eda_height = 0
        self._last_seg_width = 0
        self._last_seg_height = 0
        
        self.setup_eda_tab()
        self.setup_comparison_tab()
        self.setup_segmentation_tab()
        
    def toggle_k_slider(self):
        if self.mode_var.get() == "auto":
            self.k_slider.config(state=tk.DISABLED)
        else:
            self.k_slider.config(state=tk.NORMAL)
        
    def run_analysis(self):
        self.run_btn.config(state=tk.DISABLED)
        self.status_text.set("Running pipeline... Please wait.")
        
        # Build command based on inputs
        cmd = ["python", "main_project.py"]
        
        if self.mode_var.get() == "auto":
            cmd.append("--auto")
        else:
            cmd.extend(["--k", str(self.k_var.get())])
            
        cmd.extend(["--patch_size", str(self.patch_var.get())])
        cmd.extend(["--max_images", str(self.max_img_var.get())])
        
        print(f"Executing: {' '.join(cmd)}")
        
        def task():
            try:
                # Run the script
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.root.after(0, lambda: self.on_analysis_complete(True, result.stdout))
                else:
                    self.root.after(0, lambda: self.on_analysis_complete(False, result.stderr))
            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_complete(False, str(e)))
                
        threading.Thread(target=task, daemon=True).start()
        
    def on_analysis_complete(self, success, message):
        self.run_btn.config(state=tk.NORMAL)
        if success:
            self.status_text.set("Analysis Complete!")
            messagebox.showinfo("Success", "Pipeline finished successfully.")
            self.refresh_eda()
            self.refresh_comparison()
            self.refresh_segmentation_list()
        else:
            self.status_text.set("Analysis Failed.")
            messagebox.showerror("Error", f"Pipeline failed:\n{message}")

    # --- RESPONSIVE IMAGE HELPER ---
    def display_responsive_image(self, container, path, max_height=None):
        """Standard way to clear container and display a new resized image"""
        for w in container.winfo_children():
            w.destroy()
            
        if not os.path.exists(path):
            ttk.Label(container, text="Image not found", foreground="red").pack()
            return

        # Wait for container to have size
        self.root.update_idletasks() 
        w = container.winfo_width()
        h = max_height if max_height else container.winfo_height()
        
        # If container is too small (e.g. at startup), use defaults
        if w < 50: w = 400
        if h < 50: h = 300
            
        try:
            img = Image.open(path)
            # Calculate resize ratio to fit inside w, h while keeping aspect ratio
            img_w, img_h = img.size
            ratio = min(w / img_w, h / img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            lbl = ttk.Label(container, image=photo)
            lbl.image = photo 
            lbl.pack(expand=True)
        except Exception as e:
            ttk.Label(container, text=f"Error: {e}").pack()

    # --- EDA TAB ---
    def setup_eda_tab(self):

        self.eda_container = ttk.Frame(self.tab_eda)
        self.eda_container.pack(fill=tk.BOTH, expand=True)
        
        # 2x2 Grid or vertical stack? Let's do vertical stack with scroll
        canvas = tk.Canvas(self.eda_container)
        scrollbar = ttk.Scrollbar(self.eda_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Placeholders for images
        self.eda_plot1 = ttk.Frame(scrollable_frame)
        self.eda_plot1.pack(fill=tk.X, expand=True, pady=10)
        
        self.eda_plot2 = ttk.Frame(scrollable_frame)
        self.eda_plot2.pack(fill=tk.X, expand=True, pady=10)
        
        self.eda_plot3 = ttk.Frame(scrollable_frame)
        self.eda_plot3.pack(fill=tk.X, expand=True, pady=10)
          
        self.eda_container.destroy() # Reset
        self.setup_eda_tab_responsive()

    def setup_eda_tab_responsive(self):
        # 3 Plots: Boxplot, Correlation, Elbow.
        # Layout: Top Left (Box), Top Right (Corr), Bottom Center (Elbow)
        
        self.eda_top = ttk.Frame(self.tab_eda)
        self.eda_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.eda_bot = ttk.Frame(self.tab_eda)
        self.eda_bot.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.f_box = ttk.LabelFrame(self.eda_top, text="Distributions")
        self.f_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.f_corr = ttk.LabelFrame(self.eda_top, text="Correlations")
        self.f_corr.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.f_elbow = ttk.LabelFrame(self.eda_bot, text="Elbow Optimization")
        self.f_elbow.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind resize events
        self.tab_eda.bind("<Configure>", lambda e: self.refresh_eda())
        
        self.refresh_eda()
        
    def refresh_eda(self, event=None):
        if not self.root.geometry(): return
        
        # Prevent loop: only updates if size changed significantly
        cur_w = self.tab_eda.winfo_width()
        cur_h = self.tab_eda.winfo_height()
        
        if abs(cur_w - self._last_eda_width) < 10 and abs(cur_h - self._last_eda_height) < 10:
            return
            
        self._last_eda_width = cur_w
        self._last_eda_height = cur_h
        
        self.display_responsive_image(self.f_box, os.path.join(OUTPUT_DIR, "eda_boxplots.png"))
        self.display_responsive_image(self.f_corr, os.path.join(OUTPUT_DIR, "eda_correlation.png"))
        self.display_responsive_image(self.f_elbow, os.path.join(OUTPUT_DIR, "elbow_plot.png"))

    # --- COMPARISON TAB ---
    def setup_comparison_tab(self):
        self.comp_text = scrolledtext.ScrolledText(self.tab_comparison, font=("Consolas", 11))
        self.comp_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.refresh_comparison()
        
    def refresh_comparison(self):
        self.comp_text.delete(1.0, tk.END)
        path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                self.comp_text.insert(tk.END, f.read())
        else:
            self.comp_text.insert(tk.END, "Report not found. Please run the analysis.")

    # --- SEGMENTATION TAB ---
    def setup_segmentation_tab(self):
        # Top Controls
        control_frame = ttk.Frame(self.tab_segmentation)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Select Image:").pack(side=tk.LEFT, padx=5)
        
        self.img_var = tk.StringVar()
        self.img_combo = ttk.Combobox(control_frame, textvariable=self.img_var, state="readonly", width=50)
        self.img_combo.pack(side=tk.LEFT, padx=5)
        self.img_combo.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        ttk.Button(control_frame, text="Refresh", command=self.refresh_segmentation_list).pack(side=tk.LEFT, padx=5)
        
        # Display Area - Grid Layout for 50/50 split
        self.seg_display = ttk.Frame(self.tab_segmentation)
        self.seg_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.seg_display.columnconfigure(0, weight=1)
        self.seg_display.columnconfigure(1, weight=1)
        self.seg_display.rowconfigure(0, weight=1)
        
        self.f_orig = ttk.LabelFrame(self.seg_display, text="Original Image")
        self.f_orig.grid(row=0, column=0, sticky="nsew", padx=5)
        
        self.f_mask = ttk.LabelFrame(self.seg_display, text="Segmentation Mask")
        self.f_mask.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # State accessors
        self.current_orig_image = None
        self.current_mask_image = None
        
        # Bind resize
        self.tab_segmentation.bind("<Configure>", self.on_seg_tab_resize)
        
        self.refresh_segmentation_list()

    def refresh_segmentation_list(self):
        mask_dir = os.path.join(OUTPUT_DIR, "masks")
        if os.path.exists(mask_dir):
            files = glob.glob(os.path.join(mask_dir, "*.*"))
            basenames = [os.path.basename(f) for f in files]
            self.img_combo['values'] = basenames
            if basenames and not self.img_var.get():
                self.img_combo.current(0)
                self.on_image_selected()
        else:
            self.img_combo['values'] = []

    def on_image_selected(self, event=None):
        selected = self.img_var.get()
        if not selected: return
        
        mask_path = os.path.join(OUTPUT_DIR, "masks", selected)
        orig_path = os.path.join(DATA_DIR, selected)
        
        # Cache the PIL images so we don't re-open on every resize
        if os.path.exists(orig_path):
            self.current_orig_image = Image.open(orig_path)
        else:
            self.current_orig_image = None
            
        if os.path.exists(mask_path):
            self.current_mask_image = Image.open(mask_path)
        else:
            self.current_mask_image = None
            
        # Draw immediately
        self.redraw_segmentation_images()

    def on_seg_tab_resize(self, event):
        if event.widget != self.tab_segmentation: return
        
        # Debounce/Threshold check
        cur_w = self.tab_segmentation.winfo_width()
        cur_h = self.tab_segmentation.winfo_height()
        
        if abs(cur_w - self._last_seg_width) < 10 and abs(cur_h - self._last_seg_height) < 10:
            return
            
        self._last_seg_width = cur_w
        self._last_seg_height = cur_h
        
        self.redraw_segmentation_images()

    def redraw_segmentation_images(self):

        self.seg_display.update_idletasks()
        
        w_orig = self.f_orig.winfo_width()
        h_orig = self.f_orig.winfo_height()
        
        # If too small (e.g. init), fallback
        if w_orig < 50: w_orig = 400
        if h_orig < 50: h_orig = 300
        # Reduce padding space
        w_orig -= 20
        h_orig -= 20
        
        self.show_cached_image(self.f_orig, self.current_orig_image, w_orig, h_orig)
        self.show_cached_image(self.f_mask, self.current_mask_image, w_orig, h_orig)

    def show_cached_image(self, container, pil_image, w, h):
        # Clear old
        for widget in container.winfo_children():
            widget.destroy()
            
        if pil_image is None:
            ttk.Label(container, text="No Image", foreground="gray").pack(expand=True)
            return
            
        try:
            # Resize logic
            img_w, img_h = pil_image.size
            ratio = min(w / img_w, h / img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            
            resized = pil_image.resize(new_size, Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(resized)
            
            lbl = ttk.Label(container, image=photo)
            lbl.image = photo 
            lbl.pack(expand=True)
        except Exception as e:
            ttk.Label(container, text=f"Error: {e}").pack(expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = UrbanSegmentationApp(root)
    root.mainloop()
