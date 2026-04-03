import os
import pandas as pd
import numpy as np

INPUT_PATH = r"data/eval/eval_labels.csv"
OUTPUT_PATH = r"data/eval/eval_metrics_summary.csv"


def dcg_at_k(relevances, k):
    """
    計算 DCG@k
    """
    relevances = np.array(relevances[:k], dtype=float)

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevances) + 2))
    gains = (2 ** relevances - 1) / discounts
    return gains.sum()


def ndcg_at_k(relevances, k):
    """
    計算 NDCG@k
    """
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def precision_at_k(relevances, k, relevant_threshold=1):

    top_k = relevances[:k]

    if len(top_k) == 0:
        return 0.0

    relevant_count = sum(r >= relevant_threshold for r in top_k)
    return relevant_count / k


def hit_rate_at_k(relevances, k, relevant_threshold=1):
    """
    top-k 中只要至少有一個 relevant
    """
    top_k = relevances[:k]
    return 1.0 if any(r >= relevant_threshold for r in top_k) else 0.0


def evaluate_group(group_df, k_values=(3, 5)):
    """
    對單一 query + retrieval_type 計算 metrics
    """
    group_df = group_df.sort_values("rank")
    relevances = group_df["relevance"].tolist()

    result = {}
    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(relevances, k)
        result[f"hit_rate@{k}"] = hit_rate_at_k(relevances, k)
        result[f"ndcg@{k}"] = ndcg_at_k(relevances, k)

    return result


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到檔案: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "query_id",
        "query",
        "retrieval_type",
        "rank",
        "result_id",
        "relevance"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"eval_labels.csv 缺少以下欄位: {missing_cols}")

    # 確保型別正確
    df["rank"] = df["rank"].astype(int)
    df["relevance"] = df["relevance"].astype(int)
    df["retrieval_type"] = df["retrieval_type"].astype(str)

    all_results = []

    grouped = df.groupby(["query_id", "query", "retrieval_type"])
    for (query_id, query, retrieval_type), group_df in grouped:
        metrics = evaluate_group(group_df, k_values=(3, 5))

        row = {
            "query_id": query_id,
            "query": query,
            "retrieval_type": retrieval_type,
            **metrics
        }
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # 各 retrieval_type 平均結果
    summary_df = (
        results_df
        .groupby("retrieval_type")[["precision@3", "precision@5", "hit_rate@3", "hit_rate@5", "ndcg@3", "ndcg@5"]]
        .mean()
        .reset_index()
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("每個 query 的評估結果")
    print("=" * 100)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("各 retrieval_type 的平均結果")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print(f"\n評估結果已輸出到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()