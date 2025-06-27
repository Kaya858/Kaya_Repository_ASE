import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RangeSlider
import os
import math

# --- 科学技術計算ライブラリ ---
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

class FWHMAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("FWHM 解析プログラム - 完成版")
        master.geometry("800x600")

        self.excel_file_path = None
        self.df_results = None
        self.spectrum_files_data = []
        self.fwhm_results = {}
        self.output_graph_dir = None

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, pady=10)
        control_frame.pack(fill=tk.X)

        self.btn_load_excel = tk.Button(control_frame, text="① Excelファイルを選択", command=self.load_excel_file)
        self.btn_load_excel.pack(side=tk.LEFT, padx=10)
        
        self.btn_save = tk.Button(control_frame, text="③ 結果を保存", command=self.save_results, state="disabled")
        self.btn_save.pack(side=tk.RIGHT, padx=10)

        self.status_label = tk.Label(control_frame, text="処理を開始してください。")
        self.status_label.pack(side=tk.LEFT, padx=10)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_excel_file(self):
        self.excel_file_path = filedialog.askopenfilename(title="結果ファイル (..._after.xlsx) を選択", filetypes=[("Excel files", "*.xlsx")])
        if not self.excel_file_path: return
        try:
            self.df_results = pd.read_excel(self.excel_file_path)
            
            # グラフ保存用フォルダを作成
            output_dir = os.path.dirname(self.excel_file_path)
            self.output_graph_dir = os.path.join(output_dir, "FWHM_Analysis_Graphs")
            os.makedirs(self.output_graph_dir, exist_ok=True)
            
            self.status_label.config(text=f"'{os.path.basename(self.excel_file_path)}' を読み込みました。次に対応するスペクトルファイルを選択してください。")
            self.btn_load_excel.config(state="disabled")
            self.process_spectra()
        except Exception as e:
            messagebox.showerror("エラー", f"Excelファイルの読み込みに失敗しました: {e}")

    def process_spectra(self):
        num_spectra = len(self.df_results)
        if num_spectra == 0: return
            
        for index, row in self.df_results.iterrows():
            filter_num = row.get('filter_number', index + 1)
            spec_path = filedialog.askopenfilename(
                title=f"【{index + 1}/{num_spectra}】filter_number '{filter_num}' のスペクトルファイルを選択",
                filetypes=[("Text files", "*.txt *.txp *.csv")]
            )
            if not spec_path:
                self.status_label.config(text="処理が中断されました。")
                return
            
            try:
                spec_data = np.loadtxt(spec_path, skiprows=1)
                self.spectrum_files_data.append({'path': spec_path, 'data': spec_data, 'filter_num': filter_num})
                self.add_spectrum_graph(spec_data, f"Filter: {filter_num}", index)
            except Exception as e:
                messagebox.showerror("エラー", f"'{os.path.basename(spec_path)}' の読み込みに失敗: {e}")
                return

        self.status_label.config(text="全てのグラフを表示しました。各グラフの解析を開始してください。")
        self.btn_save.config(state="normal")

    def add_spectrum_graph(self, data, title, graph_index):
        graph_frame = tk.Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        graph_frame.pack(pady=10, padx=10, fill=tk.X)

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

        button_frame = tk.Frame(graph_frame)
        button_frame.pack(side=tk.RIGHT, padx=10, anchor='center')
        
        analyze_button = tk.Button(button_frame, text="② 解析開始", command=lambda idx=graph_index: self.start_analysis_for(idx))
        analyze_button.pack(pady=5)
        
        status_label = tk.Label(button_frame, text="未処理", fg="red")
        status_label.pack(pady=5)
        graph_frame.status_label = status_label

    def start_analysis_for(self, graph_index):
        spectrum_info = self.spectrum_files_data[graph_index]
        raw_data = spectrum_info['data']
        filter_num = spectrum_info['filter_num']
        
        # グラフ画像の保存パスを作成
        graph_save_path = os.path.join(self.output_graph_dir, f"filter_{filter_num}_result.png")
        
        dialog = AnalysisWindow(self.master, raw_data, f"詳細解析: グラフ {graph_index + 1}", graph_save_path)
        self.master.wait_window(dialog.top)

        graph_frame = self.scrollable_frame.winfo_children()[graph_index]
        if dialog.fwhm_result is not None and not np.isnan(dialog.fwhm_result):
            self.fwhm_results[graph_index] = dialog.fwhm_result
            status_text = f"FWHM: {dialog.fwhm_result:.3f}"
            status_color = "green"
        else:
            self.fwhm_results[graph_index] = np.nan
            status_text = "スキップ"
            status_color = "orange"
            
        graph_frame.status_label.config(text=status_text, fg=status_color)

    def save_results(self):
        if self.df_results is None: return
        
        base_name, _ = os.path.splitext(os.path.basename(self.excel_file_path))
        initial_file = f"{base_name.replace('_after', '')}_after_after.xlsx"

        output_path = filedialog.asksaveasfilename(
            title="結果を保存",
            filetypes=[("Excel files", "*.xlsx")],
            defaultextension=".xlsx",
            initialfile=initial_file
        )
        if not output_path: return

        fwhm_series = pd.Series(self.fwhm_results, name="FWHM (nm)").reindex(self.df_results.index)
        final_df = self.df_results.join(fwhm_series)
        
        try:
            final_df.to_excel(output_path, index=False)
            messagebox.showinfo("成功", f"結果を '{os.path.basename(output_path)}' に保存しました。\nグラフ画像は 'FWHM_Analysis_Graphs' フォルダを確認してください。")
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの保存に失敗しました: {e}")

class AnalysisWindow:
    def __init__(self, parent, data, title, save_path):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("800x600")

        self.data = data
        self.save_path = save_path # グラフ画像の保存パス
        self.fwhm_result = None

        self.x, self.y_raw = self.data[:, 0], self.data[:, 1]
        
        window_size = 11 if len(self.y_raw) > 11 else len(self.y_raw) // 2 * 2 + 1 
        if window_size < 3: window_size = 3
        self.y_smooth = savgol_filter(self.y_raw, window_size, 2)

        frame = tk.Frame(self.top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.line_raw, = self.ax.plot(self.x, self.y_raw, '.', color='skyblue', markersize=3, label="Raw Data")
        self.line_smooth, = self.ax.plot(self.x, self.y_smooth, color='blue', linewidth=1.5, label="Smoothed Data")
        self.line_fit, = self.ax.plot(self.x, self.y_smooth, '--', color='red', linewidth=2, label="Fit (Pseudo-Voigt)")
        self.fwhm_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.fig.tight_layout()

        self.slider_ax = self.fig.add_axes([0.15, 0.02, 0.65, 0.03])
        self.slider = RangeSlider(self.slider_ax, "Fit Range", self.x.min(), self.x.max())
        self.slider.on_changed(self.update_fit)
        
        button_frame = tk.Frame(self.top)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="この結果を承認", command=self.on_approve).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="スキップ", command=self.on_skip).pack(side=tk.LEFT, padx=10)
        
        self.update_fit((self.x.min(), self.x.max()))

    def pseudo_voigt(self, x, amplitude, center, fwhm, eta):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian_part = np.exp(-0.5 * ((x - center) / sigma)**2)
        lorentzian_part = 1 / (1 + ((x - center) / (fwhm / 2))**2)
        return amplitude * (eta * lorentzian_part + (1 - eta) * gaussian_part)
        
    def update_fit(self, val):
        min_val, max_val = val
        
        idx_range = (self.x >= min_val) & (self.x <= max_val)
        x_fit, y_fit = self.x[idx_range], self.y_smooth[idx_range]
        
        if len(x_fit) < 5: return
        
        try:
            peak_idx = np.argmax(y_fit)
            amplitude0 = y_fit[peak_idx]
            center0 = x_fit[peak_idx]
            fwhm0 = (x_fit[-1] - x_fit[0]) / 2.0
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
            self.fwhm_result = None

        self.canvas.draw_idle()

    def on_approve(self):
        if self.fwhm_result is None or np.isnan(self.fwhm_result):
            messagebox.showwarning("承認不可", "FWHMが有効に計算できていません。", parent=self.top)
            return
        self.save_figure()
        self.top.destroy()

    def on_skip(self):
        self.fwhm_result = None
        self.save_figure() # スキップした場合も、その時点のグラフを保存
        self.top.destroy()
        
    def save_figure(self):
        try:
            # 凡例などをつけたままの状態で保存
            self.fig.savefig(self.save_path, dpi=150)
            print(f"グラフを保存しました: {self.save_path}")
        except Exception as e:
            messagebox.showwarning("グラフ保存エラー", f"グラフの保存に失敗しました:\n{e}", parent=self.top)

if __name__ == "__main__":
    root = tk.Tk()
    app = FWHMAnalyzerApp(root)
    root.mainloop()