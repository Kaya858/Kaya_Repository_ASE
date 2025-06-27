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
from scipy.special import wofz


class FWHMAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("FWHM 解析プログラム - Step 2: 対話的フィッティング")
        master.geometry("800x600")

        self.df_results = None
        self.spectrum_files_data = []
        self.fwhm_results = {}  # FWHMの結果を格納する辞書 {index: value}

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, pady=10)
        control_frame.pack(fill=tk.X)

        self.btn_load_excel = tk.Button(control_frame, text="① Excelファイルを選択", command=self.load_excel_file)
        self.btn_load_excel.pack(side=tk.LEFT, padx=10)
        
        # 結果保存ボタン（初期状態は無効）
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
        file_path = filedialog.askopenfilename(title="結果ファイル (..._after.xlsx) を選択", filetypes=[("Excel files", "*.xlsx")])
        if not file_path: return
        try:
            self.df_results = pd.read_excel(file_path)
            self.status_label.config(text=f"'{os.path.basename(file_path)}' を読み込みました。次に対応するスペクトルファイルを選択してください。")
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
                # ユーザー仕様に合わせてヘッダーをスキップ (仮に1行とする)
                spec_data = np.loadtxt(spec_path, skiprows=1)
                self.spectrum_files_data.append({'path': spec_path, 'data': spec_data})
                self.add_spectrum_graph(spec_data, f"Filter: {filter_num}", index)
            except Exception as e:
                messagebox.showerror("エラー", f"'{os.path.basename(spec_path)}' の読み込みに失敗: {e}")
                return

        self.status_label.config(text="全てのグラフを表示しました。各グラフの解析を開始してください。")
        self.btn_save.config(state="normal") # 全て読み込んだら保存ボタンを有効化

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

        # 解析ボタンのフレーム
        button_frame = tk.Frame(graph_frame)
        button_frame.pack(side=tk.RIGHT, padx=10, anchor='center')
        
        analyze_button = tk.Button(button_frame, text="② 解析開始", command=lambda idx=graph_index: self.start_analysis_for(idx))
        analyze_button.pack(pady=5)
        
        # 解析結果を表示するラベル
        status_label = tk.Label(button_frame, text="未処理", fg="red")
        status_label.pack(pady=5)
        graph_frame.status_label = status_label # ラベルをフレームに紐付けて後でアクセス可能に

    def start_analysis_for(self, graph_index):
        raw_data = self.spectrum_files_data[graph_index]['data']
        
        # 詳細解析ウィンドウを開く
        dialog = AnalysisWindow(self.master, raw_data, f"詳細解析: グラフ {graph_index + 1}")
        self.master.wait_window(dialog.top) # ウィンドウが閉じるまで待機

        # 解析結果を反映
        if dialog.fwhm_result is not None:
            self.fwhm_results[graph_index] = dialog.fwhm_result
            status_text = f"FWHM: {dialog.fwhm_result:.2f}"
            status_color = "green"
        else:
            self.fwhm_results[graph_index] = np.nan # スキップされた場合はNaN
            status_text = "スキップ"
            status_color = "orange"
            
        # メインウィンドウのステータスラベルを更新
        graph_frame = self.scrollable_frame.winfo_children()[graph_index]
        graph_frame.status_label.config(text=status_text, fg=status_color)

    def save_results(self):
        if self.df_results is None: return
        
        output_path = filedialog.asksaveasfilename(
            title="結果を保存",
            filetypes=[("Excel files", "*.xlsx")],
            defaultextension=".xlsx",
            initialfile=f"{os.path.splitext(self.df_results.attrs.get('filename', 'result'))[0]}_after_after.xlsx"
        )
        if not output_path: return

        # FWHMの結果をDataFrameに追加
        fwhm_series = pd.Series(self.fwhm_results, name="FWHM (nm)").reindex(self.df_results.index)
        final_df = self.df_results.join(fwhm_series)
        
        try:
            final_df.to_excel(output_path, index=False)
            messagebox.showinfo("成功", f"結果を '{os.path.basename(output_path)}' に保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの保存に失敗しました: {e}")


class AnalysisWindow:
    def __init__(self, parent, data, title):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("800x600")

        self.data = data
        self.fwhm_result = None

        # --- データ前処理 ---
        self.x, self.y_raw = self.data[:, 0], self.data[:, 1]
        # Savitzky-Golayフィルターで平滑化（ウィンドウサイズと多項式の次数は調整可能）
        # ウィンドウサイズは奇数である必要がある
        window_size = 11 if len(self.y_raw) > 11 else len(self.y_raw) // 2 * 2 + 1 
        self.y_smooth = savgol_filter(self.y_raw, window_size, 3)

        # --- GUIコンポーネント ---
        frame = tk.Frame(self.top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- プロット ---
        self.line_raw, = self.ax.plot(self.x, self.y_raw, 'o', color='skyblue', markersize=2, label="Raw Data")
        self.line_smooth, = self.ax.plot(self.x, self.y_smooth, color='blue', label="Smoothed Data")
        self.line_fit, = self.ax.plot(self.x, self.y_smooth, '--', color='red', label="Fit (Pseudo-Voigt)")
        self.fwhm_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top')
        self.ax.legend()
        self.ax.grid(True)

        # --- Range Slider ---
        self.slider_ax = self.fig.add_axes([0.15, 0.02, 0.65, 0.03])
        self.slider = RangeSlider(self.slider_ax, "Fit Range", self.x.min(), self.x.max())
        self.slider.on_changed(self.update_fit)
        
        # --- ボタン ---
        button_frame = tk.Frame(self.top)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="この結果を承認", command=self.on_approve).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="スキップ", command=self.on_skip).pack(side=tk.LEFT, padx=10)
        
        self.update_fit( (self.x.min(), self.x.max()) ) # 初回描画

    # Pseudo-Voigt関数
    def pseudo_voigt(self, x, A, mu, sigma, alpha):
        # A: amplitude, mu: center, sigma: FWHM of Gaussian, alpha: mixing ratio
        sigma_g = sigma / (2 * np.sqrt(2 * np.log(2))) # Convert FWHM to std dev
        voigt_fwhm = sigma_g * 5.346 + 2 * alpha * 0.5346
        return A * ((1 - alpha) * np.exp(-0.5 * ((x - mu) / sigma_g)**2) + alpha / (1 + ((x - mu) / sigma_g)**2))
        
    def update_fit(self, val):
        min_val, max_val = val
        
        # スライダーの範囲内のデータを抽出
        idx_range = (self.x >= min_val) & (self.x <= max_val)
        x_fit = self.x[idx_range]
        y_fit = self.y_smooth[idx_range]
        
        if len(x_fit) < 4: return # フィッティングに十分なデータがない場合
        
        # --- 初期値の推定 ---
        try:
            peak_idx = np.argmax(y_fit)
            mu0 = x_fit[peak_idx]
            A0 = y_fit[peak_idx]
            sigma0 = (x_fit[-1] - x_fit[0]) / 2 # FWHMの初期値
            alpha0 = 0.5 # 混合比の初期値
            
            p0 = [A0, mu0, sigma0, alpha0]
            bounds = ([0, x_fit.min(), 0, 0], [np.inf, x_fit.max(), np.inf, 1])

            popt, _ = curve_fit(self.pseudo_voigt, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)

            # FWHMの計算
            # 簡易的にフィット曲線の半値幅を計算する
            y_fitted_full = self.pseudo_voigt(self.x, *popt)
            half_max = popt[0] / 2.0
            spline = np.interp
            try:
                # 半値になる左右のx座標を探す
                left_idx = np.where(y_fitted_full[:np.argmax(y_fitted_full)] < half_max)[0][-1]
                right_idx = np.where(y_fitted_full[np.argmax(y_fitted_full):] < half_max)[0][0] + np.argmax(y_fitted_full)
                fwhm = self.x[right_idx] - self.x[left_idx]
                self.fwhm_result = fwhm
            except IndexError: # 半値が見つからない場合
                self.fwhm_result = float('nan')

            self.line_fit.set_data(self.x, y_fitted_full)
            self.fwhm_text.set_text(f'FWHM: {self.fwhm_result:.2f} nm')
        
        except (RuntimeError, ValueError):
            self.line_fit.set_ydata(np.full_like(self.x, np.nan)) # フィット失敗時は非表示
            self.fwhm_text.set_text('FWHM: Fit Failed')
            self.fwhm_result = None

        self.canvas.draw_idle()

    def on_approve(self):
        if self.fwhm_result is None or np.isnan(self.fwhm_result):
            messagebox.showwarning("承認不可", "フィッティングに失敗しているか、FWHMが計算できていません。", parent=self.top)
            return
        self.top.destroy()

    def on_skip(self):
        self.fwhm_result = None
        self.top.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FWHMAnalyzerApp(root)
    root.mainloop()