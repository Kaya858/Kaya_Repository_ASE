import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox # エラー表示のために追加
import os
import pandas as pd
from tqdm import tqdm
import math # 円周率(pi)のために追加

# --- 設定セクション ---
TARGET_WAVELENGTH = 337
ABSORPTION_SPECTRUM_SKIP_HEADER = 55
EMISSION_SPECTRUM_SKIP_ROWS = 1
# --------------------

class AreaInputDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("励起エリア面積の計算")
        self.top.geometry("350x250") # ウィンドウサイズを少し調整
        self.result_area = None

        # --- ウィジェットの作成 ---
        self.shape_var = tk.StringVar(value="rectangle")

        tk.Label(self.top, text="励起形状を選択してください:").grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=5)
        tk.Radiobutton(self.top, text="長方形", variable=self.shape_var, value="rectangle", command=self.update_fields).grid(row=1, column=0, columnspan=2, sticky="w", padx=20)
        tk.Radiobutton(self.top, text="円形", variable=self.shape_var, value="circle", command=self.update_fields).grid(row=3, column=0, columnspan=2, sticky="w", padx=20)
        tk.Radiobutton(self.top, text="楕円形", variable=self.shape_var, value="ellipse", command=self.update_fields).grid(row=5, column=0, columnspan=2, sticky="w", padx=20)

        self.entry_rect1 = self.create_entry_row("縦 (cm):", 2, 0)
        self.entry_rect2 = self.create_entry_row("横 (cm):", 2, 2)
        self.entry_circ_d = self.create_entry_row("直径 (cm):", 4, 0)
        self.entry_elli_a = self.create_entry_row("長軸の直径 (cm):", 6, 0)
        self.entry_elli_b = self.create_entry_row("短軸の直径 (cm):", 6, 2)

        tk.Button(self.top, text="OK", command=self.on_ok, width=10).grid(row=7, column=0, columnspan=2, pady=15)
        tk.Button(self.top, text="キャンセル", command=self.on_cancel, width=10).grid(row=7, column=2, columnspan=2, pady=15)
        
        self.update_fields()

    def create_entry_row(self, label_text, row, col):
        tk.Label(self.top, text=label_text).grid(row=row, column=col, sticky="e", padx=5)
        entry = tk.Entry(self.top, width=10)
        entry.grid(row=row, column=col+1, sticky="w", padx=5)
        return entry

    def update_fields(self):
        shape = self.shape_var.get()
        for entry in [self.entry_rect1, self.entry_rect2, self.entry_circ_d, self.entry_elli_a, self.entry_elli_b]:
            entry.config(state="disabled")

        if shape == "rectangle":
            self.entry_rect1.config(state="normal")
            self.entry_rect2.config(state="normal")
        elif shape == "circle":
            self.entry_circ_d.config(state="normal")
        elif shape == "ellipse":
            self.entry_elli_a.config(state="normal")
            self.entry_elli_b.config(state="normal")
            
    def on_ok(self):
        try:
            shape = self.shape_var.get()
            if shape == "rectangle":
                a = float(self.entry_rect1.get())
                b = float(self.entry_rect2.get())
                self.result_area = a * b
            elif shape == "circle":
                d = float(self.entry_circ_d.get())
                self.result_area = math.pi * (d / 2)**2
            elif shape == "ellipse":
                a = float(self.entry_elli_a.get())
                b = float(self.entry_elli_b.get())
                self.result_area = math.pi * (a / 2) * (b / 2)
            
            if self.result_area <= 0:
                messagebox.showerror("入力エラー", "寸法は正の数値を入力してください。", parent=self.top)
                self.result_area = None
                return

            self.top.destroy()
        except ValueError:
            messagebox.showerror("入力エラー", "有効な数値を入力してください。", parent=self.top)
            self.result_area = None

    def on_cancel(self):
        self.result_area = None
        self.top.destroy()

def calculate_absorbed_energy_density(excitation_intensity, absorbance, excitation_area_size):
    absorption_rate = 1 - 10**(-absorbance)
    absorbed_energy_density = (excitation_intensity * absorption_rate) / excitation_area_size
    return absorbed_energy_density

def create_and_save_results(excitation_data, output_file, absorbance, excitation_area_size, closest_wavelength, absorption_rate, emission_intensities):
    absorbed_energy_densities = [calculate_absorbed_energy_density(row[1], absorbance, excitation_area_size) for row in excitation_data]

    df = pd.DataFrame(excitation_data, columns=['filter_number', 'excitation_intensity'])
    df['excitation_area_size(cm^2)'] = excitation_area_size
    df['closest_wavelength(nm)'] = closest_wavelength
    df['absorbance'] = absorbance
    df['absorption_rate'] = absorption_rate
    df['absorbed_energy_density'] = absorbed_energy_densities
    df['emission_intensity'] = emission_intensities
    
    try:
        df.to_excel(output_file, index=False)
        print(f"\n吸収エネルギー密度を計算し、'{output_file}' に出力しました。")
    except Exception as e:
        print(f"\nエラー: Excelファイル '{output_file}' の書き込み中にエラーが発生しました: {e}")

def extract_absorbance_from_spectrum(spectrum_file, target_wavelength, skip_header):
    try:
        spectrum_data = np.genfromtxt(spectrum_file, skip_header=skip_header, encoding='latin-1', delimiter=',', unpack=False, invalid_raise=False)
        if spectrum_data is None: raise ValueError("No data loaded.")
        if spectrum_data.ndim < 2: raise ValueError("Data needs at least 2 columns.")
        spectrum_data = spectrum_data[~np.isnan(spectrum_data).any(axis=1)]
        wavelengths, absorbances = spectrum_data[:, 0], spectrum_data[:, 1]
        idx = np.argmin(np.abs(wavelengths - target_wavelength))
        closest_wl, abs_val = wavelengths[idx], absorbances[idx]
        abs_rate = 1 - 10**(-abs_val)
        print(f"{target_wavelength} nm に最も近い波長: {closest_wl:.2f} nm, 吸光度: {abs_val:.4f}, 吸収率: {abs_rate:.4f}")
        return abs_val, closest_wl, abs_rate
    except Exception as e:
        print(f"エラー: スペクトルファイル '{spectrum_file}' の処理中にエラー: {e}")
        return None, None, None

def extract_emission_intensity(emission_file, skip_rows):
    try:
        emission_data = np.loadtxt(emission_file, skiprows=skip_rows)
        return np.max(emission_data[:, 1])
    except Exception as e:
        print(f"エラー: '{os.path.basename(emission_file)}' の処理中にエラー: {e}")
        return None

def main():
    root = tk.Tk()
    root.withdraw()

    input_file_path = filedialog.askopenfilename(title="励起強度データファイル(.xlsx)を選択してください", filetypes=[("Excel files", "*.xlsx")])
    if not input_file_path: return

    spectrum_file_path = filedialog.askopenfilename(title="吸収スペクトルファイルを選択してください", filetypes=[("Text files", "*.txt *.csv")])
    if not spectrum_file_path: return

    base_name, _ = os.path.splitext(input_file_path)
    output_file_path = f"{base_name}_after.xlsx"

    try:
        df_excitation = pd.read_excel(input_file_path, header=0)
        excitation_data = df_excitation.values
    except Exception as e:
        print(f"エラー: Excelファイル '{input_file_path}' の読み込み中にエラー: {e}")
        return

    absorbance, closest_wavelength, absorption_rate = extract_absorbance_from_spectrum(spectrum_file_path, TARGET_WAVELENGTH, ABSORPTION_SPECTRUM_SKIP_HEADER)
    if absorbance is None: return

    emission_intensities = []
    print("\n発光スペクトルファイルを選択してください...")
    for i, row in tqdm(enumerate(excitation_data), total=len(excitation_data), desc="発光スペクトル処理"):
        filter_number = int(row[0])
        emission_file_path = filedialog.askopenfilename(title=f"{filter_number}番のフィルターの発光スペクトルファイルを選択してください", filetypes=[("Text files", "*.txp")])
        if not emission_file_path:
            print("\n処理がキャンセルされました。")
            return
        
        emission_intensity = extract_emission_intensity(emission_file_path, EMISSION_SPECTRUM_SKIP_ROWS)
        if emission_intensity is None: return
        emission_intensities.append(emission_intensity)

    # --- ここで励起エリアサイズを入力 ---
    dialog = AreaInputDialog(root)
    root.wait_window(dialog.top) # 専用ウィンドウが閉じるまで待機

    if dialog.result_area is not None:
        excitation_area_size = dialog.result_area
        print(f"計算された励起エリア面積: {excitation_area_size:.4f} cm^2")
    else:
        print("\n面積計算がキャンセルされたか、エラーが発生しました。処理を中断します。")
        return

    create_and_save_results(
        excitation_data,
        output_file_path,
        absorbance,
        excitation_area_size,
        closest_wavelength,
        absorption_rate,
        np.array(emission_intensities)
    )

if __name__ == "__main__":
    main()