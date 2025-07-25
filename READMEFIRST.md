# Kaya ASE Treatment Programs
この一連のpythonプログラムはASE測定のデータを簡単に処理するためのプログラムです。(classicなレーザー測定のデータ処理にも適用可能)

このプログラムの位置付け:
測定におけるデータ取得 → 各専用ソフトウェアにおけるデータのプレ処理 → このプログラムによるデータ処理 → originによるグラフ作成およびしきい値決定

想定している計測方法および各ソフトウェアにおけるデータプレ処理の仕方、3つのプログラムそれぞれの詳細の使い方、はそれぞれ詳細なreadmeファイルが用意されているのでそれぞれ確認してね

# このリポジトリのプログラムの概要
1_Kaya_ASE_power_treatment.py, 2_Kaya_ASE_abs_E_dens.py, 3_Kaya_ASE_FWHM_fitting.py,の3つのプログラムから構成されます

このプログラムの一連のざっくりした流れ:
1. 1_Kaya_ASE_power_treatment.pyでu-joulemeterで取得したパワーデータを自動でフィッティングしてフィルタ番号とexcitation intensityが対応したエクセルファイルを作成します。
2. 2_Kaya_ASE_abs_E_dens.pyに、1のエクセルファイルと励起エリアのジオメトリとabsorption spectraを処理させることで、absorbed energy densityを算出し、そのエクセルファイルに新しい列として追加で出力します。
3. 3_Kaya_ASE_FWHM_fitting.pyでPMAスペクトロメータで取得したemissionを処理し、FWHMとpeak intensityを算出し、1,2で処理したexcelに新しい列として追記します。
これらの一連の処理をすれば、originでグラフを作るのに必要な数値はすべて揃うから、あとは簡単にグラフが書けるよ！
なお、1,2,3は独立に動作するので、プログラムに投げる前のexcelのフォーマットさえ合わせればマニュアルでexcelを作成・編集して一部だけプログラムに任せることも可能だよ

楽しいASEライフを！

