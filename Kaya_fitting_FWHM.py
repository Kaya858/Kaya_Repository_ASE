import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FWHMAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("FWHM 解析プログラム - Step 1: グラフ表示")
        master.geometry("800x600")

        # --- データ保持用変数 ---
        self.df_results = None
        self.spectrum_files_data = [] # 各スペクトルの生データを保持

        # --- メインフレーム ---
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 上部コントロールフレーム ---
        control_frame = tk.Frame(main_frame, pady=10)
        control_frame.pack(fill=tk.X)

        self.btn_load_excel = tk.Button(control_frame, text="① Excelファイルを選択", command=self.load_excel_file)
        self.btn_load_excel.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(control_frame, text="処理を開始してください。")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # --- グラフ表示用スクロールフレーム ---
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
        
        # マウスホイールでのスクロールを有効化
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)


    def _on_mousewheel(self, event):
        # WindowsやLinuxではevent.deltaが120の倍数、macOSでは1の倍数になることが多い
        # 正負を判定し、適切な単位でスクロールする
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def load_excel_file(self):
        file_path = filedialog.askopenfilename(
            title="結果ファイル (..._after.xlsx) を選択",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not file_path:
            return

        try:
            self.df_results = pd.read_excel(file_path)
            self.status_label.config(text=f"'{os.path.basename(file_path)}' を読み込みました。次に対応するスペクトルファイルを選択してください。")
            self.btn_load_excel.config(state="disabled") # ボタンを無効化
            self.process_spectra() # スペクトル処理を開始
        except Exception as e:
            self.status_label.config(text=f"エラー: Excelファイルの読み込みに失敗しました: {e}")

    def process_spectra(self):
        num_spectra = len(self.df_results)
        if num_spectra == 0:
            self.status_label.config(text="Excelファイルにデータがありません。")
            return
            
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
                self.spectrum_files_data.append(spec_data)
                
                # グラフをGUIに追加
                self.add_spectrum_graph(spec_data, f"Filter: {filter_num}", index)
                
            except Exception as e:
                self.status_label.config(text=f"エラー: '{os.path.basename(spec_path)}' の読み込みに失敗: {e}")
                return # エラーが発生したら処理を中断

        self.status_label.config(text="全てのグラフを表示しました。各グラフの解析を開始してください。")

    def add_spectrum_graph(self, data, title, graph_index):
        # 各グラフを格納するフレーム
        graph_frame = tk.Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        graph_frame.pack(pady=10, padx=10, fill=tk.X)

        # MatplotlibのFigureとAxesを作成
        fig = plt.Figure(figsize=(7, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(data[:, 0], data[:, 1])
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        fig.tight_layout()

        # Tkinterにグラフを埋め込む
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 解析開始ボタン
        # commandには後で実装する関数を紐付ける
        analyze_button = tk.Button(graph_frame, text="② 解析開始", 
                                   command=lambda idx=graph_index: self.start_analysis_for(idx))
        analyze_button.pack(side=tk.RIGHT, padx=10, pady=10, anchor='center')

    def start_analysis_for(self, graph_index):
        # この機能は第二段階で実装します
        print(f"解析開始ボタンが押されました: グラフインデックス {graph_index}")
        # ここで詳細ウィンドウを開く処理を後ほど追加します
        raw_data = self.spectrum_files_data[graph_index]
        # ... 


if __name__ == "__main__":
    import os
    root = tk.Tk()
    app = FWHMAnalyzerApp(root)
    root.mainloop()