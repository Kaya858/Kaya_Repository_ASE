import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import sv_ttk # For a modern theme

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- DETAILED ANALYSIS POP-UP WINDOW                                     ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class AnalysisWindow:
    """
    This pop-up window handles the detailed, interactive analysis of a single spectrum.
    It returns the final FWHM, intensity, and the approved graph figure itself.
    """
    def __init__(self, parent, data, title):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("800x600")
        self.top.protocol("WM_DELETE_WINDOW", self.on_skip)
        self.fwhm_result = np.nan
        self.peak_intensity_result = np.nan
        self.approved_figure = None
        self.x, self.y_raw = data[:, 0], data[:, 1]
        window_size = 11 if len(self.y_raw) > 11 else max(3, len(self.y_raw) // 2 * 2 + 1)
        self.y_smooth = savgol_filter(self.y_raw, window_size, 2)
        self._create_widgets()
        self.update_fit((self.x.min(), self.x.max()))
    def _create_widgets(self):
        frame = ttk.Frame(self.top, padding=10); frame.pack(fill=tk.BOTH, expand=True)
        self.fig = plt.Figure(); self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.x, self.y_raw, '.', color='skyblue', markersize=3, label="Raw Data")
        self.ax.plot(self.x, self.y_smooth, color='blue', linewidth=1.5, label="Smoothed Data")
        self.line_fit, = self.ax.plot([], [], '--', color='red', linewidth=2, label="Fit (Pseudo-Voigt)")
        self.fwhm_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        self.ax.legend(); self.ax.grid(True); self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Intensity")
        self.fig.tight_layout(pad=2.0)
        self.slider_ax = self.fig.add_axes([0.15, 0.02, 0.65, 0.03])
        self.slider = RangeSlider(self.slider_ax, "Fit Range", self.x.min(), self.x.max())
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.slider.on_changed(self.update_fit)
        button_frame = ttk.Frame(self.top, padding=5); button_frame.pack()
        ttk.Button(button_frame, text="Approve this Result", command=self.on_approve, style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Skip", command=self.on_skip).pack(side=tk.LEFT, padx=10)
    def pseudo_voigt(self, x, amplitude, center, fwhm, eta):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2))); gaussian_part = np.exp(-0.5 * ((x - center) / sigma)**2)
        lorentzian_part = 1 / (1 + ((x - center) / (fwhm / 2))**2)
        return amplitude * (eta * lorentzian_part + (1 - eta) * gaussian_part)
    def update_fit(self, val):
        min_val, max_val = val
        if hasattr(self, 'fit_range_span'): self.fit_range_span.remove()
        self.fit_range_span = self.ax.axvspan(min_val, max_val, color='gray', alpha=0.2, zorder=0)
        idx_range = (self.x >= min_val) & (self.x <= max_val)
        x_fit, y_fit_smooth, y_raw_in_range = self.x[idx_range], self.y_smooth[idx_range], self.y_raw[idx_range]
        if len(x_fit) < 5: self.canvas.draw_idle(); return
        try:
            peak_idx = np.argmax(y_fit_smooth)
            p0 = [y_fit_smooth[peak_idx], x_fit[peak_idx], (x_fit[-1] - x_fit[0]) / 2.0, 0.5]
            bounds = ([0, x_fit.min(), 0, 0], [np.inf, x_fit.max(), np.inf, 1])
            popt, _ = curve_fit(self.pseudo_voigt, x_fit, y_fit_smooth, p0=p0, bounds=bounds, maxfev=5000)
            self.fwhm_result = popt[2]; self.line_fit.set_data(self.x, self.pseudo_voigt(self.x, *popt))
            self.peak_intensity_result = np.max(y_raw_in_range) if y_raw_in_range.size > 0 else np.nan
            self.fwhm_text.set_text(f'FWHM: {self.fwhm_result:.3f} nm\nPeak Intensity: {self.peak_intensity_result:.2f}')
        except (RuntimeError, ValueError):
            self.line_fit.set_data([], []); self.fwhm_text.set_text('FWHM: Fit Failed\nPeak Intensity: N/A')
            self.fwhm_result, self.peak_intensity_result = np.nan, np.nan
        self.canvas.draw_idle()
    def on_approve(self):
        if np.isnan(self.fwhm_result): messagebox.showwarning("Approval Failed", "A valid FWHM has not been calculated.", parent=self.top); return
        self.approved_figure = self.fig; self.top.destroy()
    def on_skip(self):
        self.fwhm_result, self.peak_intensity_result = np.nan, np.nan; self.approved_figure = None; self.top.destroy()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- MAIN APPLICATION CLASS (Integrated)                         ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class IntegratedAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Spectroscopic Analysis Tool")
        self.root.geometry("900x950")
        self.root.minsize(750, 600)
        self.TARGET_WAVELENGTH = 337; self.ABSORPTION_SPECTRUM_SKIP_HEADER = 55; self.EMISSION_SPECTRUM_SKIP_ROWS = 1
        self._initialize_variables()
        self._initialize_data_storage()
        self._create_widgets()
        self._log("Application started. Please select files to begin.")

    def _initialize_variables(self):
        self.cut_excitation_var = tk.BooleanVar(value=False); self.lower_bound_var = tk.StringVar(); self.upper_bound_var = tk.StringVar()
        self.shape_var = tk.StringVar(value="rectangle")
        self.rect_height_var = tk.StringVar(); self.rect_width_var = tk.StringVar()
        self.circle_diameter_var = tk.StringVar(); self.ellipse_major_var = tk.StringVar(); self.ellipse_minor_var = tk.StringVar()
        self.rect_height_unit_var = tk.StringVar(value='cm'); self.rect_width_unit_var = tk.StringVar(value='cm')
        self.circle_diameter_unit_var = tk.StringVar(value='cm'); self.ellipse_major_unit_var = tk.StringVar(value='cm')
        self.ellipse_minor_unit_var = tk.StringVar(value='cm')
        self.excitation_file_path_var = tk.StringVar(value="No file selected."); self.absorption_file_path_var = tk.StringVar(value="No file selected.")
        self.emission_folder_path_var = tk.StringVar(value="No folder selected.")

    def _initialize_data_storage(self):
        self.excitation_full_path = ""; self.absorption_full_path = ""; self.emission_full_path = ""
        self.output_graph_dir = None; self.df_results = None; self.spectrum_files_data = []
        self.fwhm_results = {}; self.peak_intensity_results = {}; self.graph_figures = {}

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10); main_frame.pack(fill=tk.BOTH, expand=True)
        self._create_file_selection_group(main_frame)
        self._create_settings_group(main_frame)
        self._create_execution_group(main_frame); self._create_log_group(main_frame)
        self._create_review_group(main_frame)
        self.root.bind("<Control-o>", lambda event: self._select_excitation_file())

    def _create_file_selection_group(self, parent):
        frame = ttk.LabelFrame(parent, text="1. File Selection"); frame.pack(fill=tk.X, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Excitation Intensity Data (.xlsx):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(frame, text="Select File...", command=self._select_excitation_file).grid(row=0, column=2, padx=5)
        ttk.Label(frame, textvariable=self.excitation_file_path_var, foreground="cyan", wraplength=500).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(frame, text="Absorption Spectrum Data:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(frame, text="Select File...", command=self._select_absorption_file).grid(row=1, column=2, padx=5)
        ttk.Label(frame, textvariable=self.absorption_file_path_var, foreground="cyan", wraplength=500).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(frame, text="Emission Spectra (Folder):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(frame, text="Select Folder...", command=self._select_emission_folder).grid(row=2, column=2, padx=5)
        ttk.Label(frame, textvariable=self.emission_folder_path_var, foreground="cyan", wraplength=500).grid(row=2, column=1, sticky="ew", padx=5)

    def _create_settings_group(self, parent):
        frame = ttk.LabelFrame(parent, text="2. Analysis Settings"); frame.pack(fill=tk.X, padx=5, pady=5)
        cut_frame = ttk.Frame(frame); cut_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(cut_frame, text="Cut excitation light", variable=self.cut_excitation_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(cut_frame, text="Ignore wavelength range:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(cut_frame, textvariable=self.lower_bound_var, width=10).grid(row=1, column=1, padx=5)
        ttk.Label(cut_frame, text="nm  to").grid(row=1, column=2)
        ttk.Entry(cut_frame, textvariable=self.upper_bound_var, width=10).grid(row=1, column=3, padx=5)
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10, padx=5)
        
        # --- ★★★ MODIFIED: New layout for area shape selection ---
        area_frame = ttk.Frame(frame); area_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(area_frame, text="Excitation Area Shape:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        unit_options = ['cm', 'mm', 'µm', 'nm']

        # -- Rectangle Row --
        ttk.Radiobutton(area_frame, text="Rectangle", variable=self.shape_var, value="rectangle", command=self._update_ui_states).grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Label(area_frame, text="Height:").grid(row=1, column=1, sticky=tk.E, padx=(10, 2))
        self.rect_height_entry = ttk.Entry(area_frame, textvariable=self.rect_height_var, width=10); self.rect_height_entry.grid(row=1, column=2)
        self.rect_height_unit_combo = ttk.Combobox(area_frame, textvariable=self.rect_height_unit_var, values=unit_options, width=4, state='readonly'); self.rect_height_unit_combo.grid(row=1, column=3, padx=(2, 10))
        ttk.Label(area_frame, text="Width:").grid(row=1, column=4, sticky=tk.E, padx=(10, 2))
        self.rect_width_entry = ttk.Entry(area_frame, textvariable=self.rect_width_var, width=10); self.rect_width_entry.grid(row=1, column=5)
        self.rect_width_unit_combo = ttk.Combobox(area_frame, textvariable=self.rect_width_unit_var, values=unit_options, width=4, state='readonly'); self.rect_width_unit_combo.grid(row=1, column=6, padx=2)

        # -- Circle Row --
        ttk.Radiobutton(area_frame, text="Circle", variable=self.shape_var, value="circle", command=self._update_ui_states).grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Label(area_frame, text="Diameter:").grid(row=2, column=1, sticky=tk.E, padx=(10, 2))
        self.circle_dia_entry = ttk.Entry(area_frame, textvariable=self.circle_diameter_var, width=10); self.circle_dia_entry.grid(row=2, column=2)
        self.circle_dia_unit_combo = ttk.Combobox(area_frame, textvariable=self.circle_diameter_unit_var, values=unit_options, width=4, state='readonly'); self.circle_dia_unit_combo.grid(row=2, column=3, padx=2)
        
        # -- Ellipse Row --
        ttk.Radiobutton(area_frame, text="Ellipse", variable=self.shape_var, value="ellipse", command=self._update_ui_states).grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Label(area_frame, text="Major axis:").grid(row=3, column=1, sticky=tk.E, padx=(10, 2))
        self.ellipse_major_entry = ttk.Entry(area_frame, textvariable=self.ellipse_major_var, width=10); self.ellipse_major_entry.grid(row=3, column=2)
        self.ellipse_major_unit_combo = ttk.Combobox(area_frame, textvariable=self.ellipse_major_unit_var, values=unit_options, width=4, state='readonly'); self.ellipse_major_unit_combo.grid(row=3, column=3, padx=(2, 10))
        ttk.Label(area_frame, text="Minor axis:").grid(row=3, column=4, sticky=tk.E, padx=(10, 2))
        self.ellipse_minor_entry = ttk.Entry(area_frame, textvariable=self.ellipse_minor_var, width=10); self.ellipse_minor_entry.grid(row=3, column=5)
        self.ellipse_minor_unit_combo = ttk.Combobox(area_frame, textvariable=self.ellipse_minor_unit_var, values=unit_options, width=4, state='readonly'); self.ellipse_minor_unit_combo.grid(row=3, column=6, padx=2)

        self._update_ui_states() # Set initial state
        # --- End of modification ---

    def _create_execution_group(self, parent):
        self.execution_frame = ttk.LabelFrame(parent, text="3. Execution"); self.execution_frame.pack(fill=tk.X, padx=5, pady=5)
        self.action_frame = ttk.Frame(self.execution_frame); self.action_frame.pack(fill=tk.X, expand=True)
        self._set_initial_actions()
        self.progressbar = ttk.Progressbar(self.execution_frame, orient='horizontal', mode='determinate'); self.progressbar.pack(pady=(5, 10), padx=10, fill=tk.X)

    def _create_log_group(self, parent):
        frame = ttk.LabelFrame(parent, text="4. Log & Status"); frame.pack(fill=tk.X, padx=5, pady=5)
        self.log_text = tk.Text(frame, height=8, state=tk.DISABLED, wrap=tk.WORD, bg="gray25", fg="white")
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log_text.yview); self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _create_review_group(self, parent):
        self.review_outer_frame = ttk.LabelFrame(parent, text="5. FWHM Analysis Review")
        canvas_frame = ttk.Frame(self.review_outer_frame); canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.review_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.review_canvas.yview)
        self.review_scrollable_frame = ttk.Frame(self.review_canvas)
        self.review_scrollable_frame.bind("<Configure>", lambda e: self.review_canvas.configure(scrollregion=self.review_canvas.bbox("all")))
        self.review_canvas.create_window((0, 0), window=self.review_scrollable_frame, anchor="nw", width=800)
        self.review_canvas.configure(yscrollcommand=scrollbar.set)
        self.review_canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self.root.bind_all("<MouseWheel>", lambda e: self.review_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    
    def _set_initial_actions(self):
        for widget in self.action_frame.winfo_children(): widget.destroy()
        self.run_button = ttk.Button(self.action_frame, text="Run Analysis & Prepare Review", command=self._run_initial_analysis, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5, pady=5)

    def _set_review_actions(self):
        self.execution_frame.config(text="3. Final Actions")
        for widget in self.action_frame.winfo_children(): widget.destroy()
        self.save_button = ttk.Button(self.action_frame, text="Save Final Results", command=self._save_final_results, style="Accent.TButton")
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.reset_button = ttk.Button(self.action_frame, text="Reset All", command=self._reset_application)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def _reset_application(self):
        if not messagebox.askyesno("Confirm Reset", "Are you sure you want to clear all data and reset?"): return
        self._initialize_variables(); self._initialize_data_storage()
        self.excitation_file_path_var.set("No file selected."); self.absorption_file_path_var.set("No file selected.")
        self.emission_folder_path_var.set("No folder selected.")
        self.progressbar['value'] = 0; self.execution_frame.config(text="3. Execution")
        self._set_initial_actions()
        for widget in self.review_scrollable_frame.winfo_children(): widget.destroy()
        self.review_outer_frame.pack_forget()
        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self._log("Application has been reset.")

    def _run_initial_analysis(self):
        self.run_button.config(state=tk.DISABLED); self.progressbar['value'] = 0
        self._log("================== ANALYSIS STARTED ==================")
        try:
            self._log("Step 1/4: Validating inputs...")
            if not all([self.excitation_full_path, self.absorption_full_path, self.emission_full_path]): raise ValueError("All file/folder paths must be selected.")
            emission_files = sorted([f for f in os.listdir(self.emission_full_path) if f.lower().endswith('.txp')])
            if not emission_files: raise ValueError("No '.txp' files found in the selected folder.")
            excitation_area_size = self._calculate_area()
            self.output_graph_dir = os.path.join(os.path.dirname(self.excitation_full_path), "Analysis_Graphs"); os.makedirs(self.output_graph_dir, exist_ok=True)
            self._log("Step 2/4: Calculating energy density..."); df_excitation = pd.read_excel(self.excitation_full_path, header=0)
            absorbance, closest_wl, abs_rate = self._extract_absorbance(self.absorption_full_path)
            absorbed_energy_densities = [(row[1] * abs_rate) / excitation_area_size for row in df_excitation.values]
            self.df_results = df_excitation.copy()
            self.df_results.update({'excitation_area_size(cm^2)': excitation_area_size, 'closest_wavelength(nm)': closest_wl, 'absorbance': absorbance, 'absorption_rate': abs_rate, 'absorbed_energy_density': absorbed_energy_densities})
            self._log(f"Step 3/4: Auto-fitting {len(emission_files)} spectra for review..."); self.progressbar['maximum'] = len(emission_files)
            for i, filename in enumerate(emission_files):
                filter_num = int(os.path.splitext(filename)[0].split('_')[0])
                try: original_index = self.df_results.index[self.df_results['filter_number'] == filter_num].item()
                except (IndexError, ValueError): self._log(f"Filter number {filter_num} from '{filename}' not found in Excel. Skipping.", "WARNING"); continue
                spec_path = os.path.join(self.emission_full_path, filename); spec_data = np.loadtxt(spec_path, skiprows=self.EMISSION_SPECTRUM_SKIP_ROWS)
                fwhm, intensity, fig = self._perform_auto_fit(spec_data, f"Filter: {filter_num}")
                self.fwhm_results[original_index] = fwhm; self.peak_intensity_results[original_index] = intensity; self.graph_figures[original_index] = fig
                self.spectrum_files_data.append({'path': spec_path, 'data': spec_data, 'filter_num': filter_num, 'original_index': original_index})
                self._add_spectrum_to_review(fig, f"Filter: {filter_num}", original_index); self.progressbar.step(1)
            self._log("Step 4/4: Ready for review. Please analyze spectra below.")
            self._set_review_actions(); self.review_outer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            self._log(f"An error occurred: {e}", "ERROR"); messagebox.showerror("Error", f"An unexpected error occurred during initial analysis:\n{e}"); self.run_button.config(state=tk.NORMAL)

    def _perform_auto_fit(self, data, title):
        dialog_sim = AnalysisWindow(self.root, data, title)
        fwhm, intensity, fig = dialog_sim.fwhm_result, dialog_sim.peak_intensity_result, dialog_sim.fig
        dialog_sim.top.destroy(); return fwhm, intensity, fig
    
    def _add_spectrum_to_review(self, fig, title, index):
        graph_frame = ttk.Frame(self.review_scrollable_frame, relief=tk.RIDGE, borderwidth=2, padding=5); graph_frame.pack(pady=10, padx=10, fill=tk.X)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame); canvas.draw(); canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        button_frame = ttk.Frame(graph_frame); button_frame.pack(side=tk.RIGHT, padx=10, anchor='center')
        ttk.Button(button_frame, text="Analyze...", command=lambda idx=index: self._start_analysis_for(idx)).pack(pady=5)
        status_label = ttk.Label(button_frame, text="Ready for Review", foreground="cyan", justify=tk.LEFT, width=20); status_label.pack(pady=5)
        graph_frame.winfo_children()[1].children['!label2'] = status_label

    def _start_analysis_for(self, index):
        spectrum_info = next(item for item in self.spectrum_files_data if item['original_index'] == index)
        dialog = AnalysisWindow(self.root, spectrum_info['data'], f"Detailed Analysis: Filter {spectrum_info['filter_num']}")
        self.root.wait_window(dialog.top)
        self.fwhm_results[index] = dialog.fwhm_result; self.peak_intensity_results[index] = dialog.peak_intensity_result; self.graph_figures[index] = dialog.approved_figure
        for child_frame in self.review_scrollable_frame.winfo_children():
            try:
                title = child_frame.winfo_children()[0].figure.axes[0].get_title()
                if title == f"Filter: {spectrum_info['filter_num']}":
                    status_label = child_frame.winfo_children()[1].children['!label2']
                    if dialog.approved_figure: status_label.config(text=f"FWHM: {dialog.fwhm_result:.3f}\nPeak: {dialog.peak_intensity_result:.2f}", foreground="green")
                    else: status_label.config(text="Skipped", foreground="orange")
                    break
            except (AttributeError, KeyError): continue

    def _save_final_results(self):
        if self.df_results is None: return
        base_name, _ = os.path.splitext(os.path.basename(self.excitation_full_path))
        initial_file = f"{base_name.replace('_after', '')}_complete_analysis.xlsx"
        output_path = filedialog.asksaveasfilename(title="Save Final Results", filetypes=[("Excel files", "*.xlsx")], defaultextension=".xlsx", initialfile=initial_file)
        if not output_path: return
        final_df = self.df_results.copy()
        final_df["peak_emission_intensity"] = pd.Series(self.peak_intensity_results, name="peak_emission_intensity").reindex(final_df.index)
        final_df["FWHM (nm)"] = pd.Series(self.fwhm_results, name="FWHM (nm)").reindex(final_df.index)
        final_df.to_excel(output_path, index=False); self._log(f"Results successfully saved to '{os.path.basename(output_path)}'", "SUCCESS")
        self._log("Saving approved graph images..."); saved_count = 0
        for index, fig in self.graph_figures.items():
            if fig:
                filter_num = self.df_results.loc[index, 'filter_number']
                graph_save_path = os.path.join(self.output_graph_dir, f"filter_{filter_num}_result.png")
                fig.savefig(graph_save_path, dpi=150); saved_count += 1
        self._log(f"{saved_count} graph images saved to '{self.output_graph_dir}'.")
        messagebox.showinfo("Success", f"Analysis complete!\n\nResults saved to:\n{output_path}\n\n{saved_count} graphs saved in:\n{self.output_graph_dir}")

    def _log(self, message, level="INFO"):
        self.log_text.config(state=tk.NORMAL); self.log_text.insert(tk.END, f"[{level}] {message}\n"); self.log_text.config(state=tk.DISABLED); self.log_text.see(tk.END); self.root.update_idletasks()
    
    def _select_excitation_file(self):
        path = filedialog.askopenfilename(title="Select Excitation Intensity Data (.xlsx)", filetypes=[("Excel files", "*.xlsx *.xls")])
        if path: self.excitation_full_path = path; self.excitation_file_path_var.set(path); self._log(f"Selected excitation file: {os.path.basename(path)}")
    
    def _select_absorption_file(self):
        path = filedialog.askopenfilename(title="Select Absorption Spectrum Data", filetypes=[("Text/CSV files", "*.txt *.csv")])
        if path: self.absorption_full_path = path; self.absorption_file_path_var.set(path); self._log(f"Selected absorption file: {os.path.basename(path)}")
    
    def _select_emission_folder(self):
        path = filedialog.askdirectory(title="Select Folder Containing Emission Spectra (.txp)")
        if path: self.emission_full_path = path; self.emission_folder_path_var.set(path); self._log(f"Selected emission folder: {path}")
    
    def _update_ui_states(self):
        shape = self.shape_var.get()
        is_rect = shape == "rectangle"; is_circ = shape == "circle"; is_elli = shape == "ellipse"
        
        # Enable/Disable Rectangle widgets
        for widget in [self.rect_height_entry, self.rect_height_unit_combo, self.rect_width_entry, self.rect_width_unit_combo]:
            widget.config(state=tk.NORMAL if is_rect else tk.DISABLED)
            
        # Enable/Disable Circle widgets
        for widget in [self.circle_dia_entry, self.circle_dia_unit_combo]:
            widget.config(state=tk.NORMAL if is_circ else tk.DISABLED)
            
        # Enable/Disable Ellipse widgets
        for widget in [self.ellipse_major_entry, self.ellipse_major_unit_combo, self.ellipse_minor_entry, self.ellipse_minor_unit_combo]:
            widget.config(state=tk.NORMAL if is_elli else tk.DISABLED)

    def _convert_to_cm(self, value_str, unit):
        if not value_str: return 0.0
        conversion_factors = {'cm': 1.0, 'mm': 0.1, 'µm': 0.0001, 'nm': 1.0e-7}
        try: return float(value_str) * conversion_factors.get(unit, 1.0)
        except (ValueError, TypeError): raise ValueError(f"Invalid numeric value entered: '{value_str}'")

    def _calculate_area(self):
        shape = self.shape_var.get()
        if shape == "rectangle":
            a = self._convert_to_cm(self.rect_height_var.get(), self.rect_height_unit_var.get())
            b = self._convert_to_cm(self.rect_width_var.get(), self.rect_width_unit_var.get())
            area = a * b
        elif shape == "circle":
            d = self._convert_to_cm(self.circle_diameter_var.get(), self.circle_diameter_unit_var.get())
            area = math.pi * (d / 2)**2
        else: # Ellipse
            a = self._convert_to_cm(self.ellipse_major_var.get(), self.ellipse_major_unit_var.get())
            b = self._convert_to_cm(self.ellipse_minor_var.get(), self.ellipse_minor_unit_var.get())
            area = math.pi * (a / 2) * (b / 2)
        if area <= 0: raise ValueError("Dimensions must result in a positive area.")
        return area

    def _extract_absorbance(self, file_path):
        data = np.genfromtxt(file_path, skip_header=self.ABSORPTION_SPECTRUM_SKIP_HEADER, encoding='latin-1', delimiter=',')
        data = data[~np.isnan(data).any(axis=1)]; wavelengths, absorbances = data[:, 0], data[:, 1]
        idx = np.argmin(np.abs(wavelengths - self.TARGET_WAVELENGTH)); closest_wl, abs_val = wavelengths[idx], absorbances[idx]
        return abs_val, closest_wl, 1 - 10**(-abs_val)

if __name__ == "__main__":
    # High DPI awareness setting for Windows for a sharp UI
    if os.name == 'nt':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print(f"DPI Awareness setting failed: {e}")
    
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    app = IntegratedAnalysisApp(root)
    root.mainloop()