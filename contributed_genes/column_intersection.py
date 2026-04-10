"""
求 coefficients_all_top100.txt 文件中所有列的交集
脚本会自动检测文件所在路径（与脚本同目录）
"""

import os

import matplotlib.pyplot as plt

try:
    from upsetplot import from_memberships, UpSet
    HAS_UPSETPLOT = True
except ImportError:
    HAS_UPSETPLOT = False

def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

def find_intersection(filepath):
    """
    读取文件并求所有列的交集（即同时出现在每一列中的元素），同时返回每一列对应的集合。
    支持制表符、逗号、空格分隔
    第一行作为列名（表头），从第二行开始读取数据

    返回:
        intersection: set
        column_sets: list[set]
        column_names: list[str]
        error: str | None
    """
    # 尝试检测分隔符并读取第一行作为表头
    with open(filepath, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    
    if not header_line:
        return set(), [], [], "文件为空"
    
    # 检测分隔符
    if '\t' in header_line:
        sep = '\t'
    elif ',' in header_line:
        sep = ','
    else:
        sep = None  # 使用空格分隔
    
    # 解析表头获取列名
    if sep:
        column_names = [c.strip() for c in header_line.split(sep)]
    else:
        column_names = header_line.split()
    
    # 读取数据行（从第二行开始）
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()[1:] if line.strip()]  # 跳过第一行
    
    if not lines:
        return set(), [], column_names, "没有数据行"
    
    # 按列解析：每列是一个集合
    if sep:
        rows = [[c.strip() for c in line.split(sep)] for line in lines]
    else:
        rows = [line.split() for line in lines]
    
    # 转置：得到每列的所有元素
    num_cols = max(len(row) for row in rows) if rows else 0
    # 确保列数不超过列名数量
    num_cols = min(num_cols, len(column_names))
    
    column_sets = []
    for col_idx in range(num_cols):
        col_values = set()
        for row in rows:
            if col_idx < len(row) and row[col_idx].strip():
                col_values.add(row[col_idx].strip())
        column_sets.append(col_values)
    
    if not column_sets:
        return set(), [], column_names, "没有有效数据"
    
    # 确保列名数量与列集合数量一致
    column_names = column_names[:len(column_sets)]
    
    # 求所有列的交集
    result = set.intersection(*column_sets) if column_sets else set()
    
    return result, column_sets, column_names, None


def plot_upset_from_column_sets(column_sets, output_path, column_names=None):
    """
    根据每一列的集合绘制 UpSet 图，适合 7 列以上的集合交集可视化。
    
    参数
    ----
    column_sets : list[set]
        每一列对应的集合
    output_path : str
        输出图片的基础路径（原逻辑中为 PNG 路径），本函数会在此基础上同时保存 PNG 和 PDF
    column_names : list[str] | None
        每一列的名称
    """
    if not HAS_UPSETPLOT:
        return
    if not column_sets:
        return

    if column_names is None:
        column_names = [f"Col{i+1}" for i in range(len(column_sets))]

    # 计算所有唯一元素
    all_items = set()
    for s in column_sets:
        all_items.update(s)

    if not all_items:
        return

    # 为每个元素构建所属集合（membership）
    memberships = []
    for item in all_items:
        member_sets = [name for name, s in zip(column_names, column_sets) if item in s]
        if member_sets:
            memberships.append(member_sets)

    if not memberships:
        return

    data = from_memberships(memberships)
    upset = UpSet(data, subset_size="count", show_counts=True)
    upset.plot()
    plt.title("UpSet plot of column intersections")
    plt.tight_layout()
    
    # 调整画布大小，让图在 PDF 中更“窄”一些（单位：英寸）
    # 可根据需要微调，例如 (6, 4)、(5, 4) 等
    fig = plt.gcf()
    fig.set_size_inches(14, 6)
    
    # 兼容旧参数：output_path 仍然可以是任意后缀，这里统一处理成 png/pdf 两种格式
    base, ext = os.path.splitext(output_path)
    if not ext:  # 如果用户没有给后缀，就默认用 _upset 作为文件名后缀
        base = output_path
    
    png_path = base + ".png"
    pdf_path = base + ".pdf"

    # 保存为 PNG（紧凑边距）
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    # 同时保存为 PDF（矢量图，更适合论文排版，且紧凑边距）
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()

def main():
    script_dir = get_script_dir()
    filepath = os.path.join(script_dir, "coefficients_HPO_all_top100.txt")
    
    if not os.path.exists(filepath):
        print(f"错误：未找到文件 {filepath}")
        return
    
    result, column_sets, column_names, error = find_intersection(filepath)
    
    if error:
        print(f"错误：{error}")
        return
    
    # 输出结果
    result_sorted = sorted(result)
    print(f"所有列的交集共 {len(result_sorted)} 个元素：")
    print("-" * 40)
    for item in result_sorted:
        print(item)
    print("-" * 40)
    print(f"总计：{len(result_sorted)} 个")

    # 写入脚本所在路径下的新表格文件（制表符分隔，带表头）
    out_path = os.path.join(script_dir, "column_intersection_result.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("intersection\n")  # 表头
        for item in result_sorted:
            f.write(item + "\n")
    print(f"\n交集已保存到：{out_path}")

    # 绘制 UpSet 图（适合 7 列交集可视化）
    if HAS_UPSETPLOT and column_sets:
        # 这里传入基础路径，内部会自动保存为同名 png 和 pdf
        upset_path = os.path.join(script_dir, "column_intersection_upset")
        # 使用表格中的实际列名
        plot_upset_from_column_sets(column_sets, upset_path, column_names=column_names)
        print(f"交集 UpSet 图已保存到：{upset_path}.png 和 {upset_path}.pdf")
    elif not HAS_UPSETPLOT:
        print("\n提示：未安装 upsetplot 库，无法绘制 UpSet 图。")
        print("如需绘图，请先运行：pip install upsetplot matplotlib")

if __name__ == "__main__":
    main()
