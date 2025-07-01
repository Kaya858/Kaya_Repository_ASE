from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlibのインポート文を修正
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

# --- 定数 ---
class Config:
    DEBUG_MODE = False
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
        raw_data = pd.read_csv(filepath, sep=';', header=None, usecols=[1], names=['intensity'])['intensity']
        self.y_data_raw = raw_data.clip(lower=0)
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

    ### 変更点 ###
    # analyze_by_gradient の引数を sensitivity に変更
    def analyze_by_gradient(self, sensitivity: float) -> tuple[list, dict]:
        if self.y_data is None: raise ValueError("データが処理されていません。")

        # sensitivity (1.0-10.0) から high/low factorを計算
        # 値が大きいほど敏感（factorが小さく）になるように設定
        high_factor = 1.0 + (11.0 - sensitivity) * 0.4
        low_factor = high_factor / 3.0

        y_log = np.log1p(self.y_data)
        gradient = np.abs(np.gradient(y_log))
        log_local_mean = y_log.rolling(window=21, center=True, min_periods=1).mean()
        damping_factor = np.median(log_local_mean) * 0.5
        processed_gradient = gradient / (log_local_mean + damping_factor)
        median_grad = np.median(processed_gradient)
        mad = np.median(np.abs(processed_gradient - median_grad))
        robust_std = mad * 1.4826 + 1e-9

        # 計算したfactorで閾値を設定
        high_threshold = median_grad + high_factor * robust_std
        low_threshold = median_grad + low_factor * robust_std

        if low_threshold >= high_threshold: raise ValueError("低い閾値は高い閾値より小さく設定してください。")

        initial_labels = np.full(y_log.shape, 1, dtype=int)
        in_transition = False
        for i, grad_val in enumerate(processed_gradient):
            if not in_transition and grad_val > high_threshold: in_transition = True
            elif in_transition and grad_val < low_threshold: in_transition = False
            if in_transition: initial_labels[i] = 2

        change_points = np.where(initial_labels[:-1] != initial_labels[1:])[0] + 1
        boundaries = sorted(list(set([0] + list(change_points) + [len(y_log)])))
        initial_segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if start >= end: continue
            seg_type = 'Stair' if initial_labels[start] == 1 else 'Transition'
            initial_segments.append({'id': i, 'start': start, 'end': end, 'type': seg_type})
        
        debug_data = {"processed_gradient": processed_gradient, "high_threshold": high_threshold, "low_threshold": low_threshold, "initial_labels": initial_labels, "y_data": self.y_data, "x_data": self.x_data}
        return initial_segments, debug_data

    def run_cleanup_and_finalize(self):
        for _ in range(3):
            self._finalize_segments()
            self._cleanup_segments()
        self._finalize_segments()

    def _finalize_segments(self):
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
            seg_data = self.y_data.iloc[seg['start']:seg['end']]
            seg['avg'] = seg_data.mean() if not seg_data.empty else 0
        self._assign_nd_numbers(self.segments)
        self.boundaries = [s['start'] for s in self.segments] + ([self.segments[-1]['end']] if self.segments else [])

    def _cleanup_segments(self, min_stair_len: int = 20, min_transition_len: int = 3):
        if len(self.segments) < 2: return
        last_max_transition_size = None 
        i = 1
        while i < len(self.segments) - 1:
            prev_s, current_s, next_s = self.segments[i-1], self.segments[i], self.segments[i+1]
            if prev_s['type'] == 'Stair' and current_s['type'] == 'Transition' and next_s['type'] == 'Stair':
                change = next_s.get('avg', 0) - prev_s.get('avg', 0)
                should_merge = False
                if change <= 0: should_merge = True
                elif last_max_transition_size is not None:
                    if abs(change) < (last_max_transition_size / 4.0): should_merge = True
                
                if should_merge:
                    prev_s['end'] = next_s['end']
                    merged_data = self.y_data.iloc[prev_s['start']:prev_s['end']]
                    prev_s['avg'] = merged_data.mean() if not merged_data.empty else 0
                    self.segments.pop(i); self.segments.pop(i)
                    i = 0; continue
                else:
                    current_size = abs(change)
                    if last_max_transition_size is None: last_max_transition_size = current_size
                    else: last_max_transition_size = max(last_max_transition_size, current_size)
            i += 1
        i = 0
        while i < len(self.segments):
            seg = self.segments[i]
            seg_len = seg['end'] - seg['start']
            is_tiny = (seg['type'] == 'Stair' and seg_len < min_stair_len) or \
                      (seg['type'] == 'Transition' and seg_len < min_transition_len)
            if is_tiny and len(self.segments) > 1:
                target_idx = i - 1 if i > 0 else i + 1
                absorb_seg = self.segments[target_idx]
                absorb_seg['start'] = min(absorb_seg['start'], seg['start'])
                absorb_seg['end'] = max(absorb_seg['end'], seg['end'])
                merged_data = self.y_data.iloc[absorb_seg['start']:absorb_seg['end']]
                absorb_seg['avg'] = merged_data.mean() if not merged_data.empty else 0
                self.segments.pop(i)
                i = 0; continue
            i += 1

    def _assign_nd_numbers(self, segments):
        stair_counter = 0
        for seg in segments:
            if seg['type'] == 'Stair':
                num = self.filter_params['start'] + (stair_counter * self.filter_params['step'])
                seg['filter_num'] = self._normalize_filter_num(num)
                stair_counter += 1
            else: seg['filter_num'] = None

    def _normalize_filter_num(self, num: int) -> int:
        if num > 0: return (num - 1) % Config.FILTER_CYCLE + 1
        return (num - 1 + Config.FILTER_CYCLE * (abs(num) // Config.FILTER_CYCLE + 1)) % Config.FILTER_CYCLE + 1

    def update_segment_boundaries(self, seg_id: int, new_start: int, new_end: int):
        if not (0 <= seg_id < len(self.segments)): return
        target_seg = self.segments[seg_id]
        target_seg['start'] = new_start
        target_seg['end'] = new_end
        if seg_id > 0: self.segments[seg_id - 1]['end'] = new_start
        if seg_id < len(self.segments) - 1: self.segments[seg_id + 1]['start'] = new_end
        self.run_cleanup_and_finalize()

# ==============================================================================
# VIEW: UIの構築と表示を担当
# ==============================================================================
class AnalysisView:
    def __init__(self, root: tk.Tk, controller):
        self.root = root
        self.controller = controller
        self.root.title("Step Data Analyzer v16.0") ### 変更点: バージョンアップ

        self.filepath_var = tk.StringVar()
        self.start_filter_var = tk.StringVar(value="1")
        self.filter_step_var = tk.StringVar(value="2")
        self.use_smoothing_var = tk.BooleanVar(value=True)
        self.smoothing_window_var = tk.StringVar(value="15")
        
        ### 変更点: 閾値関連の変数を「感度」に一本化 ###
        self.sensitivity_var = tk.DoubleVar(value=5.0) # デフォルト値は5 (1-10の範囲)
        self.sensitivity_str = tk.StringVar(value=f"{self.sensitivity_var.get():.1f}")
        
        self.edit_start_var = tk.IntVar()
        self.edit_end_var = tk.IntVar()

        self._create_main_layout()
        self._link_vars()

    ### 変更点: 変数同期のロジックを感度用に変更 ###
    def _link_vars(self):
        """感度スライダーと入力ボックスの値を同期させる"""
        def on_slider_change(val):
            self.sensitivity_str.set(f"{float(val):.1f}")

        def on_entry_change(*args):
            try:
                val = float(self.sensitivity_str.get())
                # スライダーの範囲内に値を収める
                if 1.0 <= val <= 10.0:
                    self.sensitivity_var.set(val)
            except (ValueError, TypeError):
                pass # 不正な値は無視

        self.sensitivity_var.trace_add("write", lambda *args: on_slider_change(self.sensitivity_var.get()))
        self.sensitivity_str.trace_add("write", on_entry_change)

    def _create_main_layout(self):
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill="both", expand=True, padx=10, pady=10)
        control_frame = ttk.Frame(paned_window, width=400)
        paned_window.add(control_frame, weight=1)
        self._create_control_panel(control_frame)
        graph_frame = ttk.Frame(paned_window, width=800)
        paned_window.add(graph_frame, weight=3)
        self._create_graph_panel(graph_frame)

    def _create_control_panel(self, parent: ttk.Frame):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self._create_parameter_inputs(params_frame)
        segments_frame = ttk.LabelFrame(parent, text="Detected Stair Segments", padding="10")
        segments_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._create_segment_list(segments_frame)
        self.segment_edit_frame = ttk.LabelFrame(parent, text="Segment Editor", padding="10")
        self.segment_edit_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self._create_segment_editor(self.segment_edit_frame)
        self.segment_edit_frame.grid_remove()

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
        
        filter_frame = ttk.LabelFrame(parent, text="Filter Numbering (for Stairs)", padding=5)
        filter_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10,0))
        ttk.Label(filter_frame, text="Start Number:").grid(row=0, column=0, sticky="w")
        ttk.Entry(filter_frame, textvariable=self.start_filter_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(filter_frame, text="Step:").grid(row=1, column=0, sticky="w", pady=(0,2))
        ttk.Entry(filter_frame, textvariable=self.filter_step_var, width=10).grid(row=1, column=1, sticky="w")
        
        ### 変更点: 閾値設定UIを「感度」スライダーに変更 ###
        detection_frame = ttk.LabelFrame(parent, text="Detection Settings", padding=5)
        detection_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10,0))
        detection_frame.columnconfigure(1, weight=1)

        ttk.Label(detection_frame, text="Sensitivity (鈍感 <-> 敏感):").grid(row=0, column=0, sticky="w")
        ttk.Scale(detection_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, variable=self.sensitivity_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Entry(detection_frame, textvariable=self.sensitivity_str, width=5).grid(row=0, column=2)

        ttk.Button(parent, text="Run Analysis", command=self.controller.run_analysis, style="Accent.TButton").grid(row=5, column=0, columnspan=3, pady=(15,5), sticky="ew")

    def _create_segment_editor(self, parent: ttk.Frame):
        parent.columnconfigure(1, weight=1)
        self.edit_start_label_var = tk.StringVar()
        self.edit_end_label_var = tk.StringVar()
        def update_start_label(val): self.edit_start_label_var.set(f"{int(float(val))}")
        def update_end_label(val): self.edit_end_label_var.set(f"{int(float(val))}")
        ttk.Label(parent, text="Start:").grid(row=0, column=0)
        self.start_scale = ttk.Scale(parent, variable=self.edit_start_var, orient=tk.HORIZONTAL, command=update_start_label)
        self.start_scale.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(parent, textvariable=self.edit_start_label_var, width=5).grid(row=0, column=2)
        ttk.Label(parent, text="End:").grid(row=1, column=0)
        self.end_scale = ttk.Scale(parent, variable=self.edit_end_var, orient=tk.HORIZONTAL, command=update_end_label)
        self.end_scale.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(parent, textvariable=self.edit_end_label_var, width=5).grid(row=1, column=2)
        ttk.Button(parent, text="Apply Changes", command=self.controller.apply_segment_edit).grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

    ### 変更点: パラメータ取得部分を sensitivity に対応 ###
    def get_ui_parameters(self) -> dict:
        try:
            params = {
                "filepath": self.filepath_var.get(),
                "use_smoothing": self.use_smoothing_var.get(),
                "smoothing_window": int(self.smoothing_window_var.get()),
                "sensitivity": self.sensitivity_var.get(), # sensitivity を取得
                "start_filter": int(self.start_filter_var.get()),
                "filter_step": int(self.filter_step_var.get())
            }
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
        if model.y_data_raw is not None: self.ax.plot(model.x_data, model.y_data_raw, label='Raw Data', color=Config.RAW_DATA_COLOR, alpha=0.8, zorder=1)
        if model.y_data is not None: self.ax.plot(model.x_data, model.y_data, label="Processed Data", color=Config.PROCESSED_DATA_COLOR, zorder=2)
        for seg in model.segments:
            if seg['type'] == 'Stair':
                is_selected = (seg['id'] == model.selected_segment_id)
                color = Config.STAIR_LINE_SELECTED_COLOR if is_selected else Config.STAIR_LINE_COLOR
                self.ax.hlines(y=seg['avg'], xmin=seg['start'], xmax=seg['end']-1, color=color, linestyle='-', linewidth=2.5, zorder=4)
                if seg.get('filter_num') and model.y_data is not None:
                    y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                    y_offset = y_range * 0.05
                    self.ax.text((seg['start'] + seg['end']) / 2, seg['avg'] + y_offset, f"ND:{seg['filter_num']}", ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            elif seg['type'] == 'Transition': self.ax.axvspan(seg['start'], seg['end'], color=Config.TRANSITION_AREA_COLOR, alpha=0.3, zorder=0)
        for b in model.boundaries[1:-1]: self.ax.axvline(x=b, color=Config.SEPARATOR_LINE_COLOR, linestyle='--', linewidth=1, zorder=3)
        self.ax.set_title('Step Data Analysis Results'); self.ax.set_xlabel('Data Point Index'); self.ax.set_ylabel('Intensity (μJ)')
        self.ax.legend(loc='upper right'); self.ax.grid(True, linestyle=':', alpha=0.6)
        if model.y_data is not None and not model.y_data.empty: self.ax.set_ylim(bottom=min(0, model.y_data.min() * 1.15), top=model.y_data.max() * 1.15)
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

    def update_segment_editor(self, model: AnalysisModel):
        seg_id = model.selected_segment_id
        if seg_id == -1 or not(0 <= seg_id < len(model.segments)) or model.segments[seg_id]['type'] != 'Stair':
            self.segment_edit_frame.grid_remove()
            return
        self.segment_edit_frame.grid()
        seg = model.segments[seg_id]
        min_start = model.segments[seg_id - 1]['start'] if seg_id > 0 else 0
        max_end = model.segments[seg_id + 1]['end'] if seg_id < len(model.segments) - 1 else len(model.x_data)
        self.start_scale.config(from_=min_start, to=seg['end'] - 1)
        self.end_scale.config(from_=seg['start'] + 1, to=max_end)
        self.edit_start_var.set(seg['start'])
        self.edit_end_var.set(seg['end'])
        self.edit_start_label_var.set(f"{seg['start']}")
        self.edit_end_label_var.set(f"{seg['end']}")

    def show_error(self, title: str, message: str): messagebox.showerror(title, message)
    def show_info(self, title: str, message: str): messagebox.showinfo(title, message)

# ==============================================================================
# CONTROLLER: ViewとModelを仲介
# ==============================================================================
class AnalysisController:
    def __init__(self, root: tk.Tk):
        self.model = AnalysisModel()
        self.view = AnalysisView(root, self)
        self.view.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        s = ttk.Style()
        s.configure('Accent.TButton', relief="raised", borderwidth=2, font=('calibri', 10, 'bold'))
        self.debug_fig = None

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
            
            ### 変更点: sensitivity をモデルに渡す ###
            initial_segments, debug_data = self.model.analyze_by_gradient(params['sensitivity'])
            
            if Config.DEBUG_MODE: self._show_debug_plots(debug_data)
            self.model.segments = initial_segments
            self.model.run_cleanup_and_finalize()
            self.view.redraw_plot(self.model)
            self.view.update_segment_buttons(self.model)
            self.view.update_segment_editor(self.model)
            num_stairs = len([s for s in self.model.segments if s['type'] == 'Stair'])
            self.view.show_info("Analysis Complete", f"{num_stairs} stair segments were detected.")
        except Exception as e:
            self.view.show_error("Analysis Error", f"解析中にエラーが発生しました:\n{e}")
            self.model = AnalysisModel()
            self.view.redraw_plot(self.model)
            self.view.update_segment_buttons(self.model)

    def _show_debug_plots(self, debug_data):
        if self.debug_fig is not None and plt.fignum_exists(self.debug_fig.number): plt.close(self.debug_fig)
        self.debug_fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        self.debug_fig.canvas.manager.set_window_title('Debug Plot')
        x = debug_data["x_data"]
        axes[0].plot(x, debug_data["y_data"], label="Processed Data", color=Config.PROCESSED_DATA_COLOR)
        axes[0].set_title("Processed Data"); axes[0].grid(True, linestyle=':'); axes[0].legend()
        axes[1].plot(x, debug_data["processed_gradient"], label="Processed Gradient", color="purple", alpha=0.8)
        axes[1].axhline(y=debug_data["high_threshold"], color="red", linestyle="--", label=f'High Threshold ({debug_data["high_threshold"]:.3f})')
        axes[1].axhline(y=debug_data["low_threshold"], color="orange", linestyle="--", label=f'Low Threshold ({debug_data["low_threshold"]:.3f})')
        axes[1].set_title("Processed Gradient & Thresholds"); axes[1].set_yscale('log'); axes[1].grid(True, linestyle=':'); axes[1].legend()
        axes[2].plot(x, debug_data["initial_labels"], label="1: Stair, 2: Transition", color="green", drawstyle='steps-pre')
        axes[2].set_title("Initial Labels (Before Cleanup)"); axes[2].set_yticks([1, 2]); axes[2].set_yticklabels(['Stair', 'Transition'])
        axes[2].set_xlabel("Data Point Index"); axes[2].grid(True, linestyle=':'); axes[2].legend()
        plt.tight_layout(); plt.show(block=False)

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select a Data File", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if path: self.view.filepath_var.set(path)

    def select_segment(self, seg_id: int):
        self.model.selected_segment_id = -1 if self.model.selected_segment_id == seg_id else seg_id
        self.view.redraw_plot(self.model)
        self.view.update_segment_buttons(self.model)
        self.view.update_segment_editor(self.model)

    def apply_segment_edit(self):
        seg_id = self.model.selected_segment_id
        if seg_id == -1: return
        new_start = self.view.edit_start_var.get()
        new_end = self.view.edit_end_var.get()
        if new_start >= new_end:
            self.view.show_error("Invalid Range", "Start point must be less than end point.")
            return
        self.model.update_segment_boundaries(seg_id, new_start, new_end)
        self.view.redraw_plot(self.model)
        self.view.update_segment_buttons(self.model)
        self.view.update_segment_editor(self.model)

# ==============================================================================
# アプリケーションの実行
# ==============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1400x900")
    # Sun Valley Theme (もしインストールされていれば)
    try:
        import sv_ttk
        sv_ttk.set_theme("dark") # "light" or "dark"
    except ImportError:
        pass # なければデフォルトテーマで実行
    app = AnalysisController(root)
    root.mainloop()