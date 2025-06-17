import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob
import os

def calculate_absorbed_energy_density(excitation_intensity, absorbance, excitation_area_size):
    """
    励起強度から吸収エネルギー密度を計算する関数
    """
    absorption_rate = 1 - 10**(-absorbance)
    absorbed_energy_density = (excitation_intensity * absorption_rate) / excitation_area_size
    return absorbed_energy_density

def add_absorbed_energy_density(input_file, output_file, absorbance, excitation_area_size, closest_wavelength, absorption_rate, emission_intensities):
    """
    txtファイルに吸収エネルギー密度、励起エリアサイズ、337nmに最も近い波長、吸光度、吸収率、発光強度を追加する関数
    """
    try:
        data = np.loadtxt(input_file, skiprows=1)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_file}' が見つかりません。")
        return

    absorbed_energy_densities = []
    for row in data:
        excitation_intensity = row[1]
        absorbed_energy_density = calculate_absorbed_energy_density(excitation_intensity, absorbance, excitation_area_size)
        absorbed_energy_densities.append(absorbed_energy_density)

    # 新しい列を結合
    excitation_area_sizes = np.full(len(data), excitation_area_size) # 励起エリアサイズ
    closest_wavelengths = np.full(len(data), closest_wavelength) # 最も近い波長
    absorbances = np.full(len(data), absorbance) # 吸光度
    absorption_rates_arr = np.full(len(data), absorption_rate) # 吸収率

    output_data = np.column_stack((data, excitation_area_sizes, closest_wavelengths, absorbances, absorption_rates_arr, absorbed_energy_densities, emission_intensities))
    header = 'filter_number\texcitation_intensity\texcitation_area_size(cm^2)\tclosest_wavelength(nm)\tabsorbance\tabsorption_rate\tabsorbed_energy_density\temission_intensity'
    np.savetxt(output_file, output_data, fmt='%.6f', delimiter='\t', header=header, comments='')
    print(f"吸収エネルギー密度を計算し、'{output_file}' に出力しました。")

def extract_absorbance_from_spectrum(spectrum_file, target_wavelength=337):
    """
    吸収スペクトルファイルから指定波長(nm)の吸光度を抽出する関数
    """
    try:
        # Skip the header and any non-numeric rows
        spectrum_data = np.genfromtxt(spectrum_file, skip_header=55, encoding='latin-1', delimiter=',', unpack=False, invalid_raise=False)

        # Check if spectrum_data is None
        if spectrum_data is None:
            raise ValueError("No data was loaded from the spectrum file.")

        # Check if spectrum_data has the correct dimensions before attempting to remove NaN values
        if spectrum_data.ndim >= 2:
            # Remove NaN rows
            spectrum_data = spectrum_data[~np.isnan(spectrum_data).any(axis=1)]
        
            wavelengths = spectrum_data[:, 0]
            absorbances = spectrum_data[:, 1]

            # 最も近い波長を検索
            closest_wavelength_index = np.argmin(np.abs(wavelengths - target_wavelength))
            closest_wavelength = wavelengths[closest_wavelength_index]
            absorbance = absorbances[closest_wavelength_index]
            absorption_rate = 1 - 10**(-absorbance)
            
            print(f"{target_wavelength} nm に最も近い波長: {closest_wavelength} nm, 吸光度: {absorbance}, 吸収率: {absorption_rate}")
            return absorbance, closest_wavelength, absorption_rate
        
        else:
            raise ValueError("Spectrum data must have at least two columns (wavelength and absorbance).")


    except FileNotFoundError:
        print(f"エラー: スペクトルファイル '{spectrum_file}' が見つかりません。")
        return None, None, None
    except ValueError as e:
        print(f"エラー: スペクトルファイルの読み込みまたは処理中にエラーが発生しました: {e}")
        return None, None, None

def extract_emission_intensity(emission_file):
    """
    発光スペクトルファイルからピーク強度を抽出する関数
    """
    try:
        emission_data = np.loadtxt(emission_file, skiprows=1)
        emission_intensity = np.max(emission_data[:, 1])  # 2列目の最大値をピーク強度とする
        return emission_intensity
    except FileNotFoundError:
        print(f"エラー: 発光スペクトルファイル '{emission_file}' が見つかりません。")
        return None

def main():
    """
    メイン関数
    """
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示にする

    # 入力ファイルの選択
    input_file_path = filedialog.askopenfilename(title="励起強度データファイルを選択してください", filetypes=[("Text files", "*.txt")])
    if not input_file_path:
        return  # ファイルが選択されなかった場合は終了

    # 吸収スペクトルファイルの選択
    spectrum_file_path = filedialog.askopenfilename(title="吸収スペクトルファイルを選択してください", filetypes=[("Text files", "*.txt")])
    if not spectrum_file_path:
        return  # ファイルが選択されなかった場合は終了

    # 出力ファイルの保存先と名前の指定
    output_file_path = filedialog.asksaveasfilename(title="出力テキストファイルを保存してください", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if not output_file_path:
        return  # 保存先が選択されなかった場合は終了

    # 励起強度データファイルを読み込む
    try:
        excitation_data = np.loadtxt(input_file_path, skiprows=1)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_file_path}' が見つかりません。")
        return

    emission_intensities = []
    for i, row in enumerate(excitation_data):
        filter_number = int(row[0])  # フィルタ番号を取得
        # 発光スペクトルファイルの選択
        emission_file_path = filedialog.askopenfilename(title=f"{filter_number}番のフィルターの発光スペクトルファイルを選択してください", filetypes=[("Text files", "*.txp")])
        if not emission_file_path:
            return  # ファイルが選択されなかった場合は終了
        
        emission_intensity = extract_emission_intensity(emission_file_path)
        if emission_intensity is None:
            return  # 発光強度を抽出できなかった場合は終了
        emission_intensities.append(emission_intensity)

    # 吸光度をスペクトルファイルから抽出
    absorbance, closest_wavelength, absorption_rate = extract_absorbance_from_spectrum(spectrum_file_path)
    if absorbance is None:
        return  # 吸光度が抽出できなかった場合は終了

    # 励起エリアサイズの入力
    excitation_area_size = float(input("励起エリアサイズ (cm^2) を入力してください: "))

    # 吸収エネルギー密度を追加
    add_absorbed_energy_density(input_file_path, output_file_path, absorbance, excitation_area_size, closest_wavelength, absorption_rate, np.array(emission_intensities))

if __name__ == "__main__":
    main()