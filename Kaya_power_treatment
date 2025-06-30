import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 設定項目 ---
# グラフの見た目など、お好みで変更してください
FIG_SIZE = (12, 8)  # グラフウィンドウのサイズ (横, 縦)
ORIGINAL_DATA_COLOR = 'skyblue' # 元データの線の色
AVERAGED_LINE_COLOR = 'red' # 平均化した線の色
SEPARATOR_LINE_COLOR = 'gray' # 区切り線の色
FILTER_NUMBER_COLOR = 'black' # NDフィルタ番号の色

class ParameterDialog(simpledialog.Dialog):
    """NDフィルタのパラメータを入力するためのカスタムダイアログ"""
    def body(self, master):
        tk.Label(master, text="最初のNDフィルタ番号:").grid(row=0, sticky="w")
        tk.Label(master, text="NDフィルタのステップ:").grid(row=1, sticky="w")

        self.e1 = tk.Entry(master)
        self.e2 = tk.Entry(master)

        # 初期値を入力しておく
        self.e1.insert(0, "22")
        self.e2.insert(0, "2")
        
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        
        # 最初の入力欄にフォーカスを合わせる
        return self.e1

    def apply(self):
        try:
            start_num = int(self.e1.get())
            step_num = int(self.e2.get())
            self.result = start_num, step_num
        except ValueError:
            messagebox.showwarning("入力エラー", "有効な整数を入力してください。")
            self.result = None

def get_nd_filter_number(base_num, index, step):
    """
    NDフィルタの番号を計算する関数。36の次は1になるルールを適用。
    """
    num = base_num + index * step
    if num > 0:
        return (num - 1) % 36 + 1
    else:
        while num <= 0:
            num += 36
        return num

def main():
    """
    メインの処理を行う関数
    """
    # 1. Tkinterのルートウィンドウを作成し、非表示にする
    root = tk.Tk()
    root.withdraw()

    # 2. ファイル選択ダイアログを表示して、ファイルパスを取得
    filepath = filedialog.askopenfilename(
        title="データファイルを選択してください",
        filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not filepath:
        print("ファイルが選択されませんでした。プログラムを終了します。")
        return

    # 3. GUIでパラメータを入力させる
    dialog = ParameterDialog(root, title="パラメータ入力")
    if dialog.result is None:
        print("パラメータが入力されませんでした。プログラムを終了します。")
        return
    start_filter_num, filter_step = dialog.result

    # 4. データを読み込む
    try:
        data = pd.read_csv(filepath, sep=';', header=None, usecols=[1], names=['intensity'])
        y_data = data['intensity']
        x_data = np.arange(len(y_data))
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        messagebox.showerror("ファイルエラー", f"ファイルの読み込み中にエラーが発生しました:\n{e}")
        return

    # 5. グラフとスライダーの準備
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plt.subplots_adjust(bottom=0.25)

    initial_step_size = 100
    min_step_size = 1
    max_step_size = len(y_data) // 2
    
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    step_slider = Slider(
        ax=ax_slider,
        label='1段あたりのデータ点数',
        valmin=min_step_size,
        valmax=max_step_size,
        valinit=initial_step_size,
        valstep=1
    )

    # 6. グラフを更新する関数
    def update(val):
        ax.clear()
        step_size = int(step_slider.val)

        ax.plot(x_data, y_data, label='元のデータ', color=ORIGINAL_DATA_COLOR, zorder=1)

        num_steps = len(y_data) // step_size
        for i in range(num_steps):
            start_index = i * step_size
            end_index = (i + 1) * step_size
            segment = y_data[start_index:end_index]
            
            if segment.empty: continue
            
            avg_value = segment.mean()
            
            ax.hlines(y=avg_value, xmin=start_index, xmax=end_index-1, 
                      color=AVERAGED_LINE_COLOR, linestyle='-', linewidth=2.5, zorder=3)
            
            if i > 0:
                ax.axvline(x=start_index, color=SEPARATOR_LINE_COLOR, linestyle='--', linewidth=1, zorder=2)
                
            filter_num = get_nd_filter_number(start_filter_num, i, filter_step)
            text_x = start_index + step_size / 2
            text_y = avg_value + (y_data.max() - y_data.min()) * 0.05
            ax.text(text_x, text_y, f'ND: {filter_num}', ha='center', color=FILTER_NUMBER_COLOR, 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        ax.set_title(f'データ解析結果 (1段あたり {step_size} 点)')
        ax.set_xlabel('データポイントのインデックス')
        ax.set_ylabel('光の強度 (μJ)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.set_ylim(y_data.min() * 0.95, y_data.max() * 1.1)

        fig.canvas.draw_idle()

    # 7. スライダーのイベントと関数を接続
    step_slider.on_changed(update)

    # 8. 初回描画
    update(initial_step_size)

    # 9. グラフを表示
    plt.show()

if __name__ == '__main__':
    main()