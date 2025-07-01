import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 定数 ---
class Config:
    FIG_SIZE = (10, 8)
    RAW_DATA_COLOR = 'lightgray'
    PROCESSED_DATA_COLOR = 'skyblue'
    STAIR_LINE_COLOR = 'coral'
    STAIR_LINE_SELECTED_COLOR = 'red'
    TRANSITION_AREA_COLOR = 'gray'
    SEPARATOR_LINE_COLOR = 'gray'
    LOG_CLIP_VALUE = 1e-9
    MIN_SEGMENT_LEN = 5
    FILTER_CYCLE = 36
    GRADIENT_THRESHOLD_FACTOR = 3.0 

# ==============================================================================
# MODEL: データとビジネスロジックを管理
# ==============================================================================
class AnalysisModel:
    def __init__(self):
        self.filepath = ""
        self.y_data_raw = None
        self.y_data = None
        self.x_data = None
        self.segments = []
        self.boundaries = []
        self.filter_params = {'start': 1, 'step': 1}
        self.selected_segment_id = -1

    def load_data(self, filepath: str):
        self.filepath = filepath
        self.y_data_raw = pd.read_csv(filepath, sep=';', header=None, usecols=[1], names=['intensity'])['intensity']
        self.x_data = np.arange(len(self.y_data_raw))
        self.selected_segment_id = -1
        self.segments = []
        self.boundaries = []

    def process_data(self, use_smoothing: bool, window_size: int):
        if use_smoothing and window_size >= 2:
            smoothed = self.y_data_raw.rolling(window=window_size, center=True).mean()
            self.y_data = smoothed.fillna(method='bfill').fillna(method='ffill')
        else:
            self.y_data = self.y_data_raw.copy()
            
    def _analyze_by_gradient(self, gradient_threshold_factor: float):
        """勾配（変化率）と閾値に基づいてセグメントを検出する"""
        if self.y_data is None:
            raise ValueError("データが処理されていません。")

        gradient = np.abs(np.gradient(self.y_data))
        threshold = np.median(gradient) + gradient_threshold_factor * np.std(gradient)
        labels = np.where(gradient > threshold, 'Transition', 'Stair')
        boundaries = [0] + list(np.where(np.diff(labels) != 0)[0] + 1) + [len(self.y_data)]
        boundaries = sorted(list(set(boundaries)))

        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start >= end: continue
            seg_type = labels[start]
            segments.append({'id': i, 'start': start, 'end': end, 'type': seg_type})
        
        self.segments = segments
        self._finalize_segments()

    def _analyze_by_penalty(self, params: dict):
        """従来のPenalty法でセグメントを検出する"""
        signal = np.log1p(self.y_data.clip(lower=Config.LOG_CLIP_VALUE)) if params['use_log'] else self.y_data.copy()
        
        bkps = self._analyze_with_variable_penalty(signal, params['model'], params['chunk_size'], params['start_penalty'], params['end_penalty'])
        
        boundaries = [0] + bkps
        if not boundaries or boundaries[-1] != len(signal): boundaries.append(len(signal))

        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start >= end: continue
            log_data = signal.iloc[start:end]
            segments.append({'id': i, 'start': start, 'end': end, 'log_avg': log_data.mean() if not log_data.empty else 0})

        if params['use_classification'] and len(segments) > 1:
            for seg in segments:
                seg_data = signal.iloc[seg['start']:seg['end']]
                seg['std_dev'] = seg_data.std() if len(seg_data) > 1 else 0
                if len(seg_data) > 1:
                    gradient_val = np.polyfit(np.arange(len(seg_data)), seg_data, 1)[0]
                    seg['gradient'] = np.abs(gradient_val)
                else:
                    seg['gradient'] = 0
            
            feature_matrix = pd.DataFrame([{'std_dev': s['std_dev'], 'gradient': s['gradient']} for s in segments])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(scaled_features)
            cluster_centers_std = scaler.inverse_transform(kmeans.cluster_centers_)[:, 0]
            stair_cluster_label = np.argmin(cluster_centers_std)
            
            for seg, label in zip(segments, labels):
                seg['type'] = 'Stair' if label == stair_cluster_label else 'Transition'
            if segments: segments[0]['type'] = 'Stair'
        else:
            for seg in segments: seg['type'] = 'Stair'

        self.segments = segments
        self._finalize_segments()
    
    def _finalize_segments(self):
        """セグメントの結合、ID割り当て、平均値計算などを行う共通処理"""
        i = 0
        while i < len(self.segments) - 1:
            s1, s2 = self.segments[i], self.segments[i+1]
            if s1['type'] == s2['type']:
                s1['end'] = s2['end']
                self.segments.pop(i + 1)
                continue
            i += 1
        
        for idx, seg in enumerate(self.segments):
            seg['id'] = idx
            seg['avg'] = self.y_data.iloc[seg['start']:seg['end']].mean()
        
        self._assign_nd_numbers(self.segments)
        self.boundaries = [s['start'] for s in self.segments] + ([self.segments[-1]['end']] if self.segments else [])

    def _analyze_with_variable_penalty(self, signal, model, chunk_size, start_penalty, end_penalty):
        n_samples = len(signal)
        log_start_penalty = np.log(max(start_penalty, 1e-9))
        log_end_penalty = np.log(max(end_penalty, 1e-9))
        multiplier_func = lambda x: np.exp(log_start_penalty + (log_end_penalty - log_start_penalty) * (x / n_samples))
        all_bkps = set()
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size + (chunk_size // 4), n_samples)
            chunk_signal = signal[start:end]
            mid_point = start + len(chunk_signal) // 2
            base_multiplier = multiplier_func(mid_point)
            local_scale_factor = chunk_signal.mean() + 1.0
            current_penalty = base_multiplier * local_scale_factor
            algo = rpt.Pelt(model=model, min_size=Config.MIN_SEGMENT_LEN).fit(chunk_signal.values)
            try:
                bkps_in_chunk = algo.predict(pen=current_penalty)
                adjusted_bkps = [b + start for b in bkps_in_chunk if b < len(chunk_signal)]
                all_bkps.update(adjusted_bkps)
            except Exception:
                continue
        return sorted(list(all_bkps))

    def _assign_nd_numbers(self, segments):
        stair_counter = 0
        for seg in segments:
            if seg['type'] == 'Stair':
                num = self.filter_params['start'] + (stair_counter * self.filter_params['step'])
                seg['filter_num'] = self._normalize_filter_num(num)
                stair_counter += 1
            else:
                seg['filter_num'] = None

    def _normalize_filter_num(self, num: int) -> int:
        if num > 0: return (num - 1) % Config.FILTER_CYCLE + 1
        return (num - 1 + Config.FILTER_CYCLE * (abs(num) // Config.FILTER_CYCLE + 1)) % Config.FILTER_CYCLE + 1

# ==============================================================================
# VIEW: UIの構築と表示を担当
# ==============================================================================
class AnalysisView:
    def __init__(self, root: tk.Tk, controller):
        self.root = root; self.controller = controller
        self.root.title("Step Data Analyzer v13.2")

        self.filepath_var = tk.StringVar()
        self.start_filter_var = tk.StringVar(value="1")
        self.filter_step_var = tk.StringVar(value="1")
        self.use_smoothing_var = tk.BooleanVar(value=True)
        self.smoothing_window_var = tk.StringVar(value="5")
        
        self.analysis_method_var = tk.StringVar(value="Gradient (Threshold)")
        
        self.gradient_factor_var = tk.StringVar(value=str(Config.GRADIENT_THRESHOLD_FACTOR))
        
        self.use_log_transform_var = tk.BooleanVar(value=True)
        self.use_classification_var = tk.BooleanVar(value=True)
        self.detection_model_var = tk.StringVar(value="l2")
        self.start_penalty_var = tk.StringVar(value="0.1")
        self.end_penalty_var = tk.StringVar(value="10.0")
        self.chunk_size_var = tk.StringVar(value="500")

        self._create_main_layout()
    
    def _create_main_layout(self):
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill="both", expand=True, padx=10, pady=10)
        control_frame = ttk.Frame(paned_window, width=400)
        self._create_control_panel(control_frame)
        paned_window.add(control_frame, weight=1)
        graph_frame = ttk.Frame(paned_window, width=800)
        self._create_graph_panel(graph_frame)
        paned_window.add(graph_frame, weight=3)
    
    def _create_control_panel(self, parent: ttk.Frame):
        parent.grid_rowconfigure(1, weight=1); parent.grid_columnconfigure(0, weight=1)
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self._create_parameter_inputs(params_frame)
        segments_frame = ttk.LabelFrame(parent, text="Detected Stair Segments", padding="10")
        segments_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._create_segment_list(segments_frame)

    def _create_parameter_inputs(self, parent: ttk.Frame):
        parent.columnconfigure(1, weight=1)
        
        ttk.Label(parent, text="Data File:").grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=self.filepath_var, state="readonly").grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(parent, text="Browse...", command=self.controller.browse_file).grid(row=1, column=2, padx=(5,0))

        gen_frame = ttk.LabelFrame(parent, text="General Settings", padding=5)
        gen_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10,0))
        ttk.Checkbutton(gen_frame, text="Apply Smoothing", variable=self.use_smoothing_var).grid(row=0, column=0, sticky="w")
        ttk.Label(gen_frame, text="Window:").grid(row=0, column=1, sticky="e")
        self.smoothing_window_entry = ttk.Entry(gen_frame, textvariable=self.smoothing_window_var, width=8)
        self.smoothing_window_entry.grid(row=0, column=2, sticky="w", padx=2)

        # ★★★ 復活させたフィルタ番号設定UI ★★★
        filter_frame = ttk.LabelFrame(parent, text="Filter Numbering (for Stairs)", padding=5)
        filter_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10,0))
        ttk.Label(filter_frame, text="Start Number:").grid(row=0, column=0, sticky="w")
        ttk.Entry(filter_frame, textvariable=self.start_filter_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(filter_frame, text="Step:").grid(row=1, column=0, sticky="w", pady=(0,2))
        ttk.Entry(filter_frame, textvariable=self.filter_step_var, width=10).grid(row=1, column=1, sticky="w")
        
        method_frame = ttk.LabelFrame(parent, text="Analysis Method", padding=5)
        method_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10,0))
        method_menu = ttk.OptionMenu(method_frame, self.analysis_method_var, self.analysis_method_var.get(), 
                                     "Gradient (Threshold)", "Penalty (Ruptures)",
                                     command=lambda e: self.controller.toggle_analysis_widgets())
        method_menu.pack(fill='x')

        self.grad_frame = ttk.LabelFrame(parent, text="Gradient Method Settings", padding=5)
        self.grad_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(5,0))
        ttk.Label(self.grad_frame, text="Threshold Factor (↓ sensitive):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.grad_frame, textvariable=self.gradient_factor_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        self.pen_frame = ttk.LabelFrame(parent, text="Penalty Method Settings", padding=5)
        self.pen_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5,0))
        self.pen_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(self.pen_frame, text="Use Log Transform", variable=self.use_log_transform_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self.pen_frame, text="Model:").grid(row=1, column=0, sticky="w")
        ttk.OptionMenu(self.pen_frame, self.detection_model_var, "l2", "l2", "rbf").grid(row=1, column=1, sticky="w")
        ttk.Label(self.pen_frame, text="Start Penalty:").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.pen_frame, textvariable=self.start_penalty_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(self.pen_frame, text="End Penalty:").grid(row=3, column=0, sticky="w")
        ttk.Entry(self.pen_frame, textvariable=self.end_penalty_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(self.pen_frame, text="Chunk Size:").grid(row=4, column=0, sticky="w")
        ttk.Entry(self.pen_frame, textvariable=self.chunk_size_var, width=10).grid(row=4, column=1, sticky="w")

        ttk.Button(parent, text="Run Analysis", command=self.controller.run_analysis, style="Accent.TButton").grid(row=7, column=0, columnspan=3, pady=(15,5), sticky="ew")

    def get_ui_parameters(self) -> dict:
        try:
            params = {
                "filepath": self.filepath_var.get(), 
                "use_smoothing": self.use_smoothing_var.get(),
                "smoothing_window": int(self.smoothing_window_var.get()), 
                "analysis_method": self.analysis_method_var.get()
            }
            if params['analysis_method'] == "Gradient (Threshold)":
                params["gradient_threshold_factor"] = float(self.gradient_factor_var.get())
            else:
                params.update({
                    "use_log": self.use_log_transform_var.get(),
                    "use_classification": self.use_classification_var.get(), 
                    "model": self.detection_model_var.get(),
                    "start_penalty": float(self.start_penalty_var.get()),
                    "end_penalty": float(self.end_penalty_var.get()),
                    "chunk_size": int(self.chunk_size_var.get())
                })
            params.update({
                "start_filter": int(self.start_filter_var.get()),
                "filter_step": int(self.filter_step_var.get())
            })
            return params
        except (ValueError, TypeError) as e:
            self.show_error("Invalid Parameter", f"パラメータを確認してください。\n詳細: {e}")
            return None
    
    def _create_segment_list(self, parent: ttk.Frame):
        parent.grid_rowconfigure(0, weight=1); parent.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew"); scrollbar.grid(row=0, column=1, sticky="ns")

    def _create_graph_panel(self, parent: ttk.Frame):
        self.fig, self.ax = plt.subplots(figsize=Config.FIG_SIZE, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, parent); toolbar.update()
        self.ax.set_title('Step Data Analysis'); self.canvas.draw()

    def redraw_plot(self, model: AnalysisModel):
        self.ax.clear()
        if model.y_data_raw is not None:
            self.ax.plot(model.x_data, model.y_data_raw, label='Raw Data', color=Config.RAW_DATA_COLOR, alpha=0.8, zorder=1)
        if model.y_data is not None:
            self.ax.plot(model.x_data, model.y_data, label="Processed Data", color=Config.PROCESSED_DATA_COLOR, zorder=2)
        for seg in model.segments:
            if seg['type'] == 'Stair':
                is_selected = (seg['id'] == model.selected_segment_id)
                color = Config.STAIR_LINE_SELECTED_COLOR if is_selected else Config.STAIR_LINE_COLOR
                self.ax.hlines(y=seg['avg'], xmin=seg['start'], xmax=seg['end']-1, color=color, linestyle='-', linewidth=2.5, zorder=4)
                if seg.get('filter_num') and model.y_data is not None:
                    y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                    y_offset = y_range * 0.05
                    self.ax.text((seg['start'] + seg['end']) / 2, seg['avg'] + y_offset, f"ND:{seg['filter_num']}", ha='center',
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            elif seg['type'] == 'Transition':
                self.ax.axvspan(seg['start'], seg['end'], color=Config.TRANSITION_AREA_COLOR, alpha=0.3, zorder=0)
        for b in model.boundaries[1:-1]:
            self.ax.axvline(x=b, color=Config.SEPARATOR_LINE_COLOR, linestyle='--', linewidth=1, zorder=3)
        self.ax.set_title('Step Data Analysis Results'); self.ax.set_xlabel('Data Point Index'); self.ax.set_ylabel('Intensity (μJ)')
        self.ax.legend(loc='upper right'); self.ax.grid(True, linestyle=':', alpha=0.6)
        if model.y_data is not None and not model.y_data.empty:
            self.ax.set_ylim(bottom=min(0, model.y_data.min() * 1.15), top=model.y_data.max() * 1.15)
        self.canvas.draw_idle()

    def update_segment_buttons(self, model: AnalysisModel):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        style = ttk.Style(); style.configure("SelectedSegment.TButton", borderwidth=2, relief="sunken")
        for s in model.segments:
            if s['type'] == 'Stair':
                btn_text = f"ND: {s.get('filter_num', 'N/A'):<2} | Avg: {s['avg']:.3f} μJ"
                btn = ttk.Button(self.scrollable_frame, text=btn_text, command=lambda sid=s['id']: self.controller.select_segment(sid))
                btn.pack(fill='x', padx=5, pady=2)
                if s['id'] == model.selected_segment_id: btn.config(style="SelectedSegment.TButton")
    
    def show_error(self, title: str, message: str): messagebox.showerror(title, message)
    def show_info(self, title: str, message: str): messagebox.showinfo(title, message)

# ==============================================================================
# CONTROLLER: ViewとModelを仲介
# ==============================================================================
class AnalysisController:
    def __init__(self, root: tk.Tk):
        self.model = AnalysisModel()
        self.view = AnalysisView(root, self)
        self.toggle_analysis_widgets() 
        self.view.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        s = ttk.Style(); s.configure('Accent.TButton', font=('calibri', 10, 'bold'), foreground='white', background='#0078D7')

    def on_closing(self):
        plt.close('all')
        self.view.root.destroy()

    def run_analysis(self):
        params = self.view.get_ui_parameters()
        if params is None: return
        try:
            if not params['filepath']: raise ValueError("データファイルが選択されていません。")
            self.model.filter_params = {'start': params['start_filter'], 'step': params['filter_step']}
            self.model.load_data(params['filepath'])
            self.model.process_data(params['use_smoothing'], params['smoothing_window'])
            
            if params['analysis_method'] == "Gradient (Threshold)":
                self.model._analyze_by_gradient(params['gradient_threshold_factor'])
            else:
                self.model._analyze_by_penalty(params)

            self.view.redraw_plot(self.model)
            self.view.update_segment_buttons(self.model)
            num_stairs = len([s for s in self.model.segments if s['type'] == 'Stair'])
            self.view.show_info("Analysis Complete", f"{num_stairs} stair segments were detected.")
        except Exception as e:
            self.view.show_error("Analysis Error", f"解析中にエラーが発生しました:\n{e}")
            self.model = AnalysisModel(); self.view.redraw_plot(self.model); self.view.update_segment_buttons(self.model)

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select a Data File", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if path: self.view.filepath_var.set(path)

    def toggle_analysis_widgets(self):
        """解析手法に応じてUIの有効/無効を切り替える"""
        method = self.view.analysis_method_var.get()
        if method == "Gradient (Threshold)":
            for child in self.view.grad_frame.winfo_children():
                child.config(state="normal")
            for child in self.view.pen_frame.winfo_children():
                child.config(state="disabled")
        else: # Penalty (Ruptures)
            for child in self.view.grad_frame.winfo_children():
                child.config(state="disabled")
            for child in self.view.pen_frame.winfo_children():
                child.config(state="normal")

    def select_segment(self, seg_id: int):
        self.model.selected_segment_id = -1 if self.model.selected_segment_id == seg_id else seg_id
        self.view.redraw_plot(self.model)
        self.view.update_segment_buttons(self.model)

# ==============================================================================
# アプリケーションの実行
# ==============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1250x850")
    app = AnalysisController(root)
    root.mainloop()