import os
import pandas as pd


INPUT_PATH = r"data/raw/complaintsfull/main.csv" 
OUTPUT_PATH = r"data/processed/complaints_clean.csv"
SUMMARY_PATH = r"data/processed/data_quality_summary.txt"

# 是否刪除完全相同的 complaint text
DROP_DUPLICATE_COMPLAINT_TEXT = False

# 抽樣數量；如果不想抽樣，改成 None
SAMPLE_SIZE = 5000

# complaint text 最短長度
MIN_TEXT_LEN = 80
# =================================


def ensure_output_folder():
    os.makedirs("data/processed", exist_ok=True)


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def print_and_save(text: str, lines: list[str]):
    print(text)
    lines.append(text)


def summarize_data_quality(df: pd.DataFrame) -> list[str]:
    lines = []

    print_and_save("=" * 80, lines)
    print_and_save("1. 基本資料概況", lines)
    print_and_save("=" * 80, lines)
    print_and_save(f"資料筆數: {len(df):,}", lines)
    print_and_save(f"欄位數: {df.shape[1]}", lines)
    print_and_save(f"欄位名稱: {df.columns.tolist()}", lines)

    print_and_save("\n" + "=" * 80, lines)
    print_and_save("2. 缺失值統計", lines)
    print_and_save("=" * 80, lines)

    missing_count = df.isna().sum().sort_values(ascending=False)
    missing_ratio = (df.isna().mean() * 100).sort_values(ascending=False)

    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_ratio_pct": missing_ratio.round(2)
    })

    print_and_save(missing_df.to_string(), lines)

    print_and_save("\n" + "=" * 80, lines)
    print_and_save("3. 重複值統計", lines)
    print_and_save("=" * 80, lines)

    full_row_duplicates = df.duplicated().sum()
    print_and_save(f"完全重複列數: {full_row_duplicates:,}", lines)

    if "complaint id" in df.columns:
        complaint_id_dup = df["complaint id"].duplicated().sum()
        print_and_save(f"'complaint id' 重複數: {complaint_id_dup:,}", lines)

    if "consumer complaint narrative" in df.columns:
        temp = df["consumer complaint narrative"].astype(str).str.strip()
        temp = temp[temp != ""]
        temp = temp[temp.str.lower() != "nan"]
        text_dup = temp.duplicated().sum()
        print_and_save(f"'consumer complaint narrative' 完全重複數: {text_dup:,}", lines)

        print_and_save("\n文字長度統計（針對非空 complaint narrative）:", lines)
        text_len = temp.str.len()
        print_and_save(f"非空筆數: {len(temp):,}", lines)
        print_and_save(f"平均長度: {text_len.mean():.2f}", lines)
        print_and_save(f"中位數長度: {text_len.median():.2f}", lines)
        print_and_save(f"最短長度: {text_len.min()}", lines)
        print_and_save(f"最長長度: {text_len.max()}", lines)

    return lines


def clean_data(
    df: pd.DataFrame,
    min_text_len: int = 80,
    sample_size: int | None = 5000,
    drop_duplicate_complaint_text: bool = False
) -> pd.DataFrame:
    # 先只保留第一版需要的欄位
    keep_cols = [
        "product",
        "sub-product",
        "issue",
        "sub-issue",
        "consumer complaint narrative",
        "company public response",
        "company",
        "company response to consumer",
        "complaint id"
    ]

    missing_keep_cols = [col for col in keep_cols if col not in df.columns]
    if missing_keep_cols:
        raise ValueError(f"以下必要欄位不存在: {missing_keep_cols}")

    df = df[keep_cols].copy()

    # 先移除完全重複列
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n已移除完全重複列: {before - after:,}")

    # complaint text 清理
    text_col = "consumer complaint narrative"
    df[text_col] = df[text_col].astype(str).str.strip()

    # 核心欄位：缺了就刪
    core_cols = ["product", "issue", "complaint id", "consumer complaint narrative"]

    # 先刪真正 NaN
    df = df.dropna(subset=core_cols)

    # 再刪空字串 / nan 字串
    df = df[df["product"].astype(str).str.strip() != ""]
    df = df[df["issue"].astype(str).str.strip() != ""]
    df = df[df["complaint id"].astype(str).str.strip() != ""]
    df = df[df[text_col] != ""]
    df = df[df[text_col].str.lower() != "nan"]

    # complaint text 太短直接刪
    before = len(df)
    df = df[df[text_col].str.len() >= min_text_len].copy()
    after = len(df)
    print(f"已移除 complaint text 長度 < {min_text_len} 的資料: {before - after:,}")

    if drop_duplicate_complaint_text:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col]).copy()
        after = len(df)
        print(f"已移除完全重複 complaint text: {before - after:,}")

    # 輔助欄位：可補 Unknown
    auxiliary_cols = [
        "sub-product",
        "sub-issue",
        "company public response",
        "company",
        "company response to consumer"
    ]

    for col in auxiliary_cols:
        df[col] = df[col].fillna("Unknown")
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"
        df.loc[df[col].str.lower() == "nan", col] = "Unknown"

    # complaint id 統一轉乾淨字串
    df["complaint id"] = (
        df["complaint id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # 建立可檢索文本
    def build_page_content(row):
        parts = [
            f"Product: {row['product']}",
            f"Sub-product: {row['sub-product']}",
            f"Issue: {row['issue']}",
            f"Sub-issue: {row['sub-issue']}",
            f"Complaint: {row['consumer complaint narrative']}",
        ]
        return "\n".join(parts)

    df["page_content"] = df.apply(build_page_content, axis=1)

    # 整理最終輸出欄位
    out = pd.DataFrame({
        "complaint_id": df["complaint id"],
        "product": df["product"],
        "sub_product": df["sub-product"],
        "issue": df["issue"],
        "sub_issue": df["sub-issue"],
        "company": df["company"],
        "company_public_response": df["company public response"],
        "company_response_to_consumer": df["company response to consumer"],
        "complaint_text": df["consumer complaint narrative"],
        "page_content": df["page_content"]
    })

    # 抽樣
    if sample_size is not None:
        sample_n = min(sample_size, len(out))
        out = out.sample(n=sample_n, random_state=42).copy()
        print(f"已抽樣 {sample_n:,} 筆資料作為第一版實驗")

    return out


def save_summary(lines: list[str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_output_folder()

    print("開始讀取資料...")
    df = load_data(INPUT_PATH)

    # 先做資料品質摘要
    summary_lines = summarize_data_quality(df)

    print("\n" + "=" * 80)
    print("4. 開始清理資料")
    print("=" * 80)

    clean_df = clean_data(
        df=df,
        min_text_len=MIN_TEXT_LEN,
        sample_size=SAMPLE_SIZE,
        drop_duplicate_complaint_text=DROP_DUPLICATE_COMPLAINT_TEXT
    )

    clean_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    save_summary(summary_lines, SUMMARY_PATH)

    print("\n" + "=" * 80)
    print("5. 清理完成")
    print("=" * 80)
    print(f"輸出檔案: {OUTPUT_PATH}")
    print(f"摘要報告: {SUMMARY_PATH}")
    print(f"最終筆數: {len(clean_df):,}")
    print("\n前 3 筆資料:")
    print(clean_df.head(3).T)


if __name__ == "__main__":
    main()