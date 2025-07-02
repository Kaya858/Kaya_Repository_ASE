import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider
import os
import math
import sv_ttk  # Import the theme library

# --- Scientific Computing Libraries ---
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

class FWHMAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("FWHM Analysis Program")
        master.geometry("850x650")

        # --- Application State Variables ---
        self.excel_file_path = None
        self.df_results = None
        self.spectrum_files_data = []
        self.fwhm_results = {}
        self.output_graph_dir = None
        self.graph_frames = {}

        # --- Main UI Frames ---
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X)

        # --- Control Widgets ---
        # Highlight the primary action button
        self.btn_load_excel = ttk.Button(control_frame, text="① Select Excel File", command=self.load_excel_file, style="Accent.TButton")
        self.btn_load_excel.pack(side=tk.LEFT, padx=5)

        self.btn_reset = ttk.Button(control_frame, text="Reset", command=self._reset_application)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        # Highlight the final action button
        self.btn_save = ttk.Button(control_frame, text="③ Save Results", command=self.save_results, state="disabled", style="Accent.TButton")
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # --- Canvas for scrollable graphs ---
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Status Bar and Progress Bar ---
        status_frame = ttk.Frame(master, padding="2 5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(status_frame, text="Please start by selecting an Excel file.")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progressbar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate')
        self.progressbar.pack(side=tk.RIGHT, padx=5)

        # --- Bindings ---
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)
        self.master.bind("<Control-o>", lambda event: self.load_excel_file())
        self.master.bind("<Control-s>", lambda event: self.save_results() if self.btn_save['state'] == 'normal' else None)
        self.master.bind("<Control-r>", lambda event: self._reset_application())

    def _reset_application(self):
        """ Resets the application to its initial state. """
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to clear all data and reset the application?"):
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            self.excel_file_path = None
            self.df_results = None
            self.spectrum_files_data.clear()
            self.fwhm_results.clear()
            self.output_graph_dir = None
            self.graph_frames.clear()
            self.status_label.config(text="Please start by selecting an Excel file.")
            self.btn_load_excel.config(state="normal")
            self.btn_save.config(state="disabled")
            self.progressbar.config(value=0)
            print("Application has been reset.")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_excel_file(self):
        self.excel_file_path = filedialog.askopenfilename(title="Select the results file (..._after.xlsx)", filetypes=[("Excel files", "*.xlsx")])
        if not self.excel_file_path: return
        try:
            self.df_results = pd.read_excel(self.excel_file_path)
            output_dir = os.path.dirname(self.excel_file_path)
            self.output_graph_dir = os.path.join(output_dir, "FWHM_Analysis_Graphs")
            os.makedirs(self.output_graph_dir, exist_ok=True)
            self.status_label.config(text=f"Loaded '{os.path.basename(self.excel_file_path)}'. Please select the corresponding spectrum files.")
            self.btn_load_excel.config(state="disabled")
            self.process_spectra()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel file: {e}")

    def _ask_resolve_duplicate(self, filter_num, path1, path2):
        dialog = tk.Toplevel(self.master)
        dialog.title("Resolve Duplicate File")
        dialog.geometry("500x200")
        dialog.transient(self.master)
        dialog.grab_set()
        result = tk.StringVar(value="")
        message = f"Filter number '{filter_num}' is duplicated.\nWhich file would you like to use?"
        ttk.Label(dialog, text=message, padding=20).pack()
        frame = ttk.Frame(dialog)
        frame.pack(pady=10)
        basename1 = os.path.basename(path1)
        basename2 = os.path.basename(path2)
        def select_and_close(path):
            result.set(path)
            dialog.destroy()
        ttk.Button(frame, text=basename1, command=lambda: select_and_close(path1), width=40).pack(pady=5)
        ttk.Button(frame, text=basename2, command=lambda: select_and_close(path2), width=40).pack(pady=5)
        self.master.wait_window(dialog)
        return result.get()

    def process_spectra(self):
        num_spectra_expected = len(self.df_results)
        if num_spectra_expected == 0: return
        spec_paths = filedialog.askopenfilenames(
            title="Select spectrum files",
            filetypes=[("Spectrum files", "*.txp *.txt *.csv"), ("All files", "*.*")]
        )
        if not spec_paths:
            self.status_label.config(text="Operation cancelled.")
            self.btn_load_excel.config(state="normal")
            return
        path_map = {}
        for path in spec_paths:
            try:
                basename = os.path.basename(path)
                stem = os.path.splitext(basename)[0]
                filter_num_str = stem.split('_')[0]
                filter_num = int(filter_num_str)
                if filter_num in path_map:
                    chosen_path = self._ask_resolve_duplicate(filter_num, path_map[filter_num], path)
                    if not chosen_path:
                        messagebox.showwarning("Cancelled", "Duplicate file resolution was cancelled. Aborting process.")
                        self.btn_load_excel.config(state="normal")
                        return
                    path_map[filter_num] = chosen_path
                else:
                    path_map[filter_num] = path
            except (ValueError, IndexError):
                print(f"Info: Skipping '{basename}' as it does not match the naming convention.")
                continue
        required_filters = self.df_results.get('filter_number', pd.Series(range(1, num_spectra_expected + 1), index=self.df_results.index))
        missing_filters = [f for f in required_filters if f not in path_map]
        if missing_filters:
            msg = "The following required files were not found:\n\n" + ", ".join(map(str, missing_filters))
            msg += "\n\nDo you want to continue? (Missing data will be skipped)"
            if not messagebox.askyesno("Missing Files Confirmation", msg):
                self.status_label.config(text="Operation cancelled.")
                self.btn_load_excel.config(state="normal")
                return
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.spectrum_files_data.clear()
        self.graph_frames.clear()
        self.progressbar.config(maximum=num_spectra_expected, value=0)
        for index, row in self.df_results.iterrows():
            filter_num = row.get('filter_number', index + 1)
            spec_path = path_map.get(filter_num)
            self.progressbar.step(1)
            self.master.update_idletasks()
            if not spec_path: continue
            try:
                spec_data = np.loadtxt(spec_path, skiprows=1)
                self.spectrum_files_data.append({'path': spec_path, 'data': spec_data, 'filter_num': filter_num, 'original_index': index})
                self.add_spectrum_graph(spec_data, f"Filter: {filter_num}", index)
            except Exception as e:
                messagebox.showerror("File Read Error", f"Failed to read '{os.path.basename(spec_path)}': {e}")
                return
        self.status_label.config(text="All graphs displayed. Please begin analysis.")
        self.btn_save.config(state="normal")

    def add_spectrum_graph(self, data, title, graph_index):
        graph_frame = ttk.Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=2, padding=5)
        graph_frame.pack(pady=10, padx=10, fill=tk.X)
        
        with plt.style.context("default"):
            fig = plt.Figure(figsize=(7, 3), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(data[:, 0], data[:, 1])
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity")
            ax.grid(True)
            fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(graph_frame)
        button_frame.pack(side=tk.RIGHT, padx=10, anchor='center')
        
        # Highlight the analysis button
        analyze_button = ttk.Button(button_frame, text="② Start Analysis", command=lambda idx=graph_index: self.start_analysis_for(idx), style="Accent.TButton")
        analyze_button.pack(pady=5)
        
        status_label = ttk.Label(button_frame, text="Not Processed", foreground="red")
        status_label.pack(pady=5)
        self.graph_frames[graph_index] = {'frame': graph_frame, 'status_label': status_label}

    def start_analysis_for(self, graph_index):
        try:
            spectrum_info = next(item for item in self.spectrum_files_data if item['original_index'] == graph_index)
            status_label = self.graph_frames[graph_index]['status_label']
        except (StopIteration, KeyError):
            messagebox.showerror("Internal Error", f"Could not find data or graph for index '{graph_index}'.")
            return
        raw_data = spectrum_info['data']
        filter_num = spectrum_info['filter_num']
        graph_save_path = os.path.join(self.output_graph_dir, f"filter_{filter_num}_result.png")
        dialog = AnalysisWindow(self.master, raw_data, f"Detailed Analysis: Filter {filter_num}", graph_save_path)
        self.master.wait_window(dialog.top)
        if dialog.fwhm_result is not None and not np.isnan(dialog.fwhm_result):
            self.fwhm_results[graph_index] = dialog.fwhm_result
            status_text = f"FWHM: {dialog.fwhm_result:.3f}"
            status_color = "green"
        else:
            self.fwhm_results[graph_index] = np.nan
            status_text = "Skipped"
            status_color = "orange"
        status_label.config(text=status_text, foreground=status_color)

    def save_results(self):
        if self.df_results is None: return
        base_name, _ = os.path.splitext(os.path.basename(self.excel_file_path))
        initial_file = f"{base_name.replace('_after', '')}_after_after.xlsx"
        output_path = filedialog.asksaveasfilename(
            title="Save Results",
            filetypes=[("Excel files", "*.xlsx")],
            defaultextension=".xlsx",
            initialfile=initial_file
        )
        if not output_path: return
        fwhm_series = pd.Series(self.fwhm_results, name="FWHM (nm)").reindex(self.df_results.index)
        final_df = self.df_results.join(fwhm_series)
        try:
            final_df.to_excel(output_path, index=False)
            messagebox.showinfo("Success", f"Results saved to '{os.path.basename(output_path)}'.\nGraph images are in the 'FWHM_Analysis_Graphs' folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

class AnalysisWindow:
    def __init__(self, parent, data, title, save_path):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("800x600")
        self.top.protocol("WM_DELETE_WINDOW", self.on_skip)

        self.data = data
        self.save_path = save_path
        self.fwhm_result = np.nan
        self.fit_range_span = None

        self.x, self.y_raw = self.data[:, 0], self.data[:, 1]
        window_size = 11 if len(self.y_raw) > 11 else max(3, len(self.y_raw) // 2 * 2 + 1)
        self.y_smooth = savgol_filter(self.y_raw, window_size, 2)

        frame = ttk.Frame(self.top, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        with plt.style.context("default"):
            self.fig = plt.Figure()
            self.ax = self.fig.add_subplot(111)
            self.line_raw, = self.ax.plot(self.x, self.y_raw, '.', color='skyblue', markersize=3, label="Raw Data")
            self.line_smooth, = self.ax.plot(self.x, self.y_smooth, color='blue', linewidth=1.5, label="Smoothed Data")
            self.line_fit, = self.ax.plot(self.x, self.y_smooth, '--', color='red', linewidth=2, label="Fit (Pseudo-Voigt)")
            self.fwhm_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
            self.ax.legend()
            self.ax.grid(True)
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Intensity")
            self.fig.tight_layout(pad=2.0)
            self.slider_ax = self.fig.add_axes([0.15, 0.02, 0.65, 0.03])
            self.slider = RangeSlider(self.slider_ax, "Fit Range", self.x.min(), self.x.max())
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.slider.on_changed(self.update_fit)

        button_frame = ttk.Frame(self.top, padding=5)
        button_frame.pack()
        
        # Highlight the approval button
        approve_button = ttk.Button(button_frame, text="Approve this Result", command=self.on_approve, style="Accent.TButton")
        approve_button.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Skip", command=self.on_skip).pack(side=tk.LEFT, padx=10)
        
        self.update_fit((self.x.min(), self.x.max()))

    def pseudo_voigt(self, x, amplitude, center, fwhm, eta):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian_part = np.exp(-0.5 * ((x - center) / sigma)**2)
        lorentzian_part = 1 / (1 + ((x - center) / (fwhm / 2))**2)
        return amplitude * (eta * lorentzian_part + (1 - eta) * gaussian_part)
        
    def update_fit(self, val):
        min_val, max_val = val
        
        if self.fit_range_span:
            self.fit_range_span.remove()
        self.fit_range_span = self.ax.axvspan(min_val, max_val, color='gray', alpha=0.2, zorder=0)

        idx_range = (self.x >= min_val) & (self.x <= max_val)
        x_fit, y_fit = self.x[idx_range], self.y_smooth[idx_range]
        
        if len(x_fit) < 5: return
        
        try:
            peak_idx = np.argmax(y_fit)
            amplitude0 = y_fit[peak_idx]
            center0 = x_fit[peak_idx]
            try:
                half_max = amplitude0 / 2.0
                left_idx = (np.abs(y_fit[:peak_idx] - half_max)).argmin()
                right_idx = (np.abs(y_fit[peak_idx:] - half_max)).argmin() + peak_idx
                fwhm0 = x_fit[right_idx] - x_fit[left_idx]
            except (ValueError, IndexError):
                fwhm0 = (x_fit[-1] - x_fit[0]) / 2.0
            if fwhm0 <= 0: fwhm0 = (x_fit[-1] - x_fit[0]) / 2.0
            eta0 = 0.5
            p0 = [amplitude0, center0, fwhm0, eta0]
            bounds = ([0, x_fit.min(), 0, 0], [np.inf, x_fit.max(), np.inf, 1])
            popt, _ = curve_fit(self.pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)
            
            self.fwhm_result = popt[2]
            y_fitted_full = self.pseudo_voigt(self.x, *popt)
            self.line_fit.set_ydata(y_fitted_full)
            self.fwhm_text.set_text(f'FWHM: {self.fwhm_result:.3f} nm')
        
        except (RuntimeError, ValueError):
            self.line_fit.set_ydata(np.full_like(self.x, np.nan))
            self.fwhm_text.set_text('FWHM: Fit Failed')
            self.fwhm_result = np.nan

        self.canvas.draw_idle()

    def on_approve(self):
        if self.fwhm_result is None or np.isnan(self.fwhm_result):
            messagebox.showwarning("Approval Failed", "A valid FWHM has not been calculated.", parent=self.top)
            return
        self.save_figure()
        self.top.destroy()

    def on_skip(self):
        self.fwhm_result = np.nan
        self.save_figure()
        self.top.destroy()
        
    def save_figure(self):
        try:
            self.fig.savefig(self.save_path, dpi=150)
            print(f"Graph saved to: {self.save_path}")
        except Exception as e:
            messagebox.showwarning("Graph Save Error", f"Failed to save graph:\n{e}", parent=self.top)

if __name__ == "__main__":
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    app = FWHMAnalyzerApp(root)
    root.mainloop()