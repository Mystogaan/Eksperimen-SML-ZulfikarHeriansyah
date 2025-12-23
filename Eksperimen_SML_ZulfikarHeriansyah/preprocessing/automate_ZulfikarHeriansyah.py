import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_path: str) -> str:
    """
    Preprocessing dataset credit scoring (credit_score.csv):
    - Membaca data RAW
    - Drop kolom ID: CUST_ID (jika ada)
    - Pisahkan fitur (X) dan target (DEFAULT)
    - One-hot encoding untuk fitur kategorikal
    - Standard scaling untuk semua fitur (setelah encoding)
    - Simpan hasil preprocessing ke CSV

    Args:
        input_path (str): path ke dataset RAW (CSV)
        output_path (str): path output dataset preprocessing (CSV)

    Returns:
        str: output_path (lokasi file hasil preprocessing)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)

    # Drop kolom ID (kalau ada)
    if "CUST_ID" in df.columns:
        df = df.drop(columns=["CUST_ID"])

    # Target
    target_col = "DEFAULT"
    if target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan. Kolom yang ada: {list(df.columns)[:10]} ..."
        )

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encoding untuk fitur kategorikal
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Kembalikan ke DataFrame
    processed_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    processed_df[target_col] = y.values

    # Pastikan folder output ada
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Simpan
    processed_df.to_csv(output_path, index=False)

    return output_path


def main():
    # Lokasi file ini (preprocessing/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Root project = satu level di atas folder preprocessing/
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # Path RAW dan output preprocessing (sesuai struktur ketentuan)
    input_csv = os.path.join(project_root, "dataset_raw", "credit_score.csv")
    output_csv = os.path.join(current_dir, "dataset_preprocessing", "credit_score_preprocessed.csv")

    print("== Preprocessing Otomatis ==")
    print("Project root :", project_root)
    print("Input (RAW)  :", input_csv)
    print("Output (Prep):", output_csv)

    saved_path = preprocess_data(input_csv, output_csv)
    print(f"\nSUKSES ✅ Dataset preprocessing tersimpan di:\n{saved_path}")


if __name__ == "__main__":
    main()
