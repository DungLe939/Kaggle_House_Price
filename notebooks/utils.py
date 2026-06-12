import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_vs_target(df, features, target, cols=2):
    """
    Hàm vẽ các biểu đồ scatter plot giữa danh sách các đặc trưng và biến mục tiêu.
    - df: DataFrame dữ liệu (ví dụ: train_df)
    - features: Danh sách các cột cần vẽ (ví dụ: ['GrLivArea', 'TotalBsmtSF'])
    - target: Tên cột biến mục tiêu cần so sánh (mặc định: 'SalePrice')
    - cols: Số cột biểu đồ trên 1 hàng (mặc định: 2)
    """
    n_features = len(features)

    rows = math.ceil(n_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    
    if n_features == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    colors = sns.color_palette("Set2", n_features)
    
    for i, feature in enumerate(features):
        ax = axes_flat[i]
        sns.scatterplot(x=df[feature], y=df[target], ax=ax, color=colors[i], alpha=0.7)
        ax.set_title(f"{feature} vs {target}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{feature}", fontsize=12)
        ax.set_ylabel(f"{target}", fontsize=12)
        
    for j in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    plt.tight_layout()
    plt.show()

def plot_histograms(df, features, cols=2, bins=50):
    """
    Hàm vẽ các biểu đồ histogram kết hợp KDE của danh sách các đặc trưng.
    - df: DataFrame dữ liệu (ví dụ: train_df)
    - features: Danh sách các cột cần vẽ (ví dụ: ['GrLivArea', 'LotArea'])
    - cols: Số cột biểu đồ trên 1 hàng (mặc định: 2)
    - bins: Số lượng cột chia nhỏ (mặc định: 50)
    """
    n_features = len(features)
    rows = math.ceil(n_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    
    if n_features == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    colors = sns.color_palette("Set2", n_features)
    
    for i, feature in enumerate(features):
        ax = axes_flat[i]
        # Loại bỏ các giá trị NaN tạm thời khi vẽ để tránh lỗi cảnh báo
        sns.histplot(df[feature].dropna(), bins=bins, kde=True, ax=ax, color=colors[i])
        ax.set_title(f"Distribution of {feature}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{feature}", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        
    for j in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    plt.tight_layout()
    plt.show()


def audit_categorical_features(df, missing_threshold=0.80, dominance_threshold=0.95, unique_threshold=15):
    """
    Quét tất cả cột categorical (object) và cảnh báo 3 vấn đề:
      1. 🔴 Missing quá nhiều  (count / tổng dòng < 1 - missing_threshold)
      2. 🟡 Near-zero variance  (freq / count > dominance_threshold)
      3. 🟠 Unique quá cao      (unique > unique_threshold → OHE sẽ nổ cột)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần kiểm tra.
    missing_threshold : float, default=0.80
        Nếu tỷ lệ missing > ngưỡng này → cảnh báo "Missing cao".
    dominance_threshold : float, default=0.95
        Nếu giá trị phổ biến nhất chiếm > ngưỡng này → cảnh báo "Near-zero variance".
    unique_threshold : int, default=15
        Nếu số giá trị unique > ngưỡng này → cảnh báo "Unique cao".

    Returns
    -------
    pd.DataFrame
        Bảng tổng hợp với các cột: count, missing%, unique, top, freq, dominance%, flags.

    Example
    -------
    report = audit_categorical_features(train_df)
    # Xem tất cả cột có vấn đề
    report[report['flags'] != '']
    # Lọc riêng cột missing cao
    report[report['flags'].str.contains('Missing')]
    """
    import pandas as pd

    cat_cols = df.select_dtypes(include='object').columns
    total_rows = len(df)

    records = []
    for col in cat_cols:
        count = df[col].count()                          # Số dòng có dữ liệu
        missing_pct = 1 - (count / total_rows)           # Tỷ lệ missing
        n_unique = df[col].nunique()                     # Số giá trị khác nhau
        value_counts = df[col].value_counts()
        top_value = value_counts.index[0]                # Giá trị phổ biến nhất
        top_freq = value_counts.iloc[0]                  # Số lần xuất hiện
        dominance_pct = top_freq / count if count > 0 else 0  # freq / count

        # === Gán flags ===
        flags = []
        if missing_pct > missing_threshold:
            flags.append(f"🔴 Missing cao ({missing_pct:.1%})")
        if dominance_pct > dominance_threshold:
            flags.append(f"🟡 Near-zero var ({top_value}={dominance_pct:.1%})")
        if n_unique > unique_threshold:
            flags.append(f"🟠 Unique cao ({n_unique})")

        records.append({
            'column': col,
            'count': count,
            'missing%': f"{missing_pct:.1%}",
            'unique': n_unique,
            'top': top_value,
            'freq': top_freq,
            'dominance%': f"{dominance_pct:.1%}",
            'flags': ' | '.join(flags)
        })

    report = pd.DataFrame(records).set_index('column')

    # In tóm tắt
    flagged = report[report['flags'] != '']
    clean = report[report['flags'] == '']

    print(f"📊 Tổng cộng: {len(cat_cols)} cột categorical")
    print(f"⚠️  Có vấn đề: {len(flagged)} cột")
    print(f"✅  Sạch:       {len(clean)} cột")
    print()

    if len(flagged) > 0:
        print("=" * 80)
        print("CÁC CỘT CÓ VẤN ĐỀ:")
        print("=" * 80)

        # Nhóm theo loại vấn đề
        missing_cols = [r for _, r in flagged.iterrows() if '🔴' in r['flags']]
        nzv_cols = [r for _, r in flagged.iterrows() if '🟡' in r['flags']]
        unique_cols = [r for _, r in flagged.iterrows() if '🟠' in r['flags']]

        if missing_cols:
            print(f"\n🔴 MISSING CAO (>{missing_threshold:.0%}) — Nên xóa hoặc đổi binary flag:")
            for r in missing_cols:
                print(f"   • {r.name}: missing {r['missing%']}, chỉ {r['count']}/{total_rows} dòng có dữ liệu")

        if nzv_cols:
            print(f"\n🟡 NEAR-ZERO VARIANCE (>{dominance_threshold:.0%} cùng 1 giá trị) — Nên xóa:")
            for r in nzv_cols:
                print(f"   • {r.name}: '{r['top']}' chiếm {r['dominance%']}")

        if unique_cols:
            print(f"\n🟠 UNIQUE QUÁ CAO (>{unique_threshold}) — Cẩn thận khi OneHotEncoder:")
            for r in unique_cols:
                print(f"   • {r.name}: {r['unique']} giá trị → OHE sẽ tạo {r['unique'] - 1} cột mới")

    return report
