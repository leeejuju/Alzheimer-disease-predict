import argparse
import time

import pandas as pd


def load_SNP(SNP_file_name):
    """加载ADNI对应的755个样本经质量控制后以某种标准筛选出的SNP的txt文件（二分类: HC vs AD）"""
    print("loading SNP data ...")
    print("正在读取", SNP_file_name, "...")

    st1 = time.time()
    SNP = pd.read_csv(SNP_file_name, header=0, sep=r"\s+")
    SNP_to_topology = SNP.copy()

    en1 = time.time()
    print(f"load执行时间为: {(en1 - st1) / 60} 分钟")

    # 选出AD_label是1和4的样本
    SNP_filtered_HC_AD = SNP[SNP[SNP.columns[-1]].isin([1.0, 4.0])]

    # 将最后一列中值为 1 的样本改为 0，值为 4 的改为 1
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 1.0,
        SNP_filtered_HC_AD.columns[-1],
    ] = 0
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 4.0,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1

    # 第一列是IID，不需要，最后一列是AD_label
    SNP = SNP_filtered_HC_AD.iloc[:, 1:-1].astype(int)
    truead = SNP_filtered_HC_AD.iloc[:, [-1]].astype(int)

    print(SNP)
    print(truead)

    return SNP, truead, SNP_to_topology


def three_load_SNP(SNP_file_name):
    """加载ADNI对应755个样本经质量控制后以某种标准筛选出的SNP的txt文件（三分类）"""
    print("loading SNP data ...")
    print("正在读取", SNP_file_name, "...")

    st1 = time.time()
    SNP = pd.read_csv(SNP_file_name, header=0, sep=r"\s+")
    SNP_to_topology = SNP.copy()

    en1 = time.time()
    print(f"load执行时间为: {(en1 - st1) / 60} 分钟")

    SNP_filtered_HC_AD = SNP[SNP[SNP.columns[-1]].isin([1, 2, 3, 4])]

    # 重编码标签: 1->0, 2->1, 3->1, 4->2
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 1,
        SNP_filtered_HC_AD.columns[-1],
    ] = 0
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 2,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 3,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 4,
        SNP_filtered_HC_AD.columns[-1],
    ] = 2

    IID = SNP_filtered_HC_AD.iloc[:, [0]].reset_index(drop=True)
    SNP = SNP_filtered_HC_AD.iloc[:, 1:-1].astype(int).reset_index(drop=True)
    truead = SNP_filtered_HC_AD.iloc[:, [-1]].astype(int).reset_index(drop=True)

    return SNP, truead, SNP_to_topology, IID


def load_Pheno(Pheno_file_name):
    """加载ADNI对应的755个样本的剔除协变量效应后的三个模态的116个脑区QT的excel文件"""
    print("loading Pheno data ...")
    print("正在读取", Pheno_file_name, "文件 ...")

    Pheno = pd.read_excel(Pheno_file_name, sheet_name=3)
    Pheno, truead = Pheno.iloc[:, 1:-1], Pheno.iloc[:, [-1]]

    print(Pheno)
    print(truead)

    return Pheno, truead


def three_load_IID(SNP_file_name):
    """从三分类SNP文件中提取样本IID"""
    SNP = pd.read_csv(SNP_file_name, header=0, sep=r"\s+")

    SNP_filtered_HC_AD = SNP[SNP[SNP.columns[-1]].isin([1, 2, 3, 4])]

    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 1,
        SNP_filtered_HC_AD.columns[-1],
    ] = 0
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 2,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 3,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 4,
        SNP_filtered_HC_AD.columns[-1],
    ] = 2

    IID = SNP_filtered_HC_AD.iloc[:, [0]].reset_index(drop=True)

    return IID


def load_IID(SNP_file_name):
    """从二分类SNP文件中提取样本IID"""
    SNP = pd.read_csv(SNP_file_name, header=0, sep=r"\s+")

    SNP_filtered_HC_AD = SNP[SNP[SNP.columns[-1]].isin([1.0, 4.0])]

    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 1.0,
        SNP_filtered_HC_AD.columns[-1],
    ] = 0
    SNP_filtered_HC_AD.loc[
        SNP_filtered_HC_AD[SNP_filtered_HC_AD.columns[-1]] == 4.0,
        SNP_filtered_HC_AD.columns[-1],
    ] = 1

    IID = SNP_filtered_HC_AD.iloc[:, [0]]

    print(IID)

    return IID


def load_Pheno_ID(Pheno_file_name):
    """从Pheno文件中提取样本IID"""
    print("loading Pheno data ...")
    print("正在读取", Pheno_file_name, "文件 ...")

    Pheno = pd.read_excel(Pheno_file_name, sheet_name=3)
    IID = Pheno.iloc[:, [0]]

    print(IID)

    return IID


# ───────────────────────── CLI 入口 ─────────────────────────


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="ADNI 阿尔兹海默症 SNP / 表型数据加载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  # 二分类方式加载SNP数据
  python Load_SNP_pheno.py load-snp --snp-file data/snp.txt

  # 三分类方式加载SNP数据
  python Load_SNP_pheno.py three-load-snp --snp-file data/snp.txt

  # 加载表型数据
  python Load_SNP_pheno.py load-pheno --pheno-file data/pheno.xlsx

  # 提取样本IID（二分类）
  python Load_SNP_pheno.py load-iid --snp-file data/snp.txt

  # 提取样本IID（三分类）
  python Load_SNP_pheno.py three-load-iid --snp-file data/snp.txt

  # 从表型文件提取IID
  python Load_SNP_pheno.py load-pheno-id --pheno-file data/pheno.xlsx
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="选择要执行的操作")
    subparsers.required = True

    # ── load-snp: 二分类加载 ──
    sp = subparsers.add_parser("load-snp", help="二分类方式加载SNP数据（HC vs AD）")
    sp.add_argument(
        "--snp-file", required=True, type=str, help="SNP 数据文件路径（txt，空格分隔）"
    )

    # ── three-load-snp: 三分类加载 ──
    sp = subparsers.add_parser("three-load-snp", help="三分类方式加载SNP数据")
    sp.add_argument(
        "--snp-file", required=True, type=str, help="SNP 数据文件路径（txt，空格分隔）"
    )

    # ── load-pheno: 加载表型 ──
    sp = subparsers.add_parser("load-pheno", help="加载表型（Phenotype）数据")
    sp.add_argument(
        "--pheno-file", required=True, type=str, help="表型数据文件路径（Excel）"
    )

    # ── load-iid: 二分类提取IID ──
    sp = subparsers.add_parser("load-iid", help="从二分类SNP文件中提取样本IID")
    sp.add_argument(
        "--snp-file", required=True, type=str, help="SNP 数据文件路径（txt，空格分隔）"
    )

    # ── three-load-iid: 三分类提取IID ──
    sp = subparsers.add_parser("three-load-iid", help="从三分类SNP文件中提取样本IID")
    sp.add_argument(
        "--snp-file", required=True, type=str, help="SNP 数据文件路径（txt，空格分隔）"
    )

    # ── load-pheno-id: 从表型文件提取IID ──
    sp = subparsers.add_parser("load-pheno-id", help="从表型文件中提取样本IID")
    sp.add_argument(
        "--pheno-file", required=True, type=str, help="表型数据文件路径（Excel）"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "load-snp": lambda: load_SNP(args.snp_file),
        "three-load-snp": lambda: three_load_SNP(args.snp_file),
        "load-pheno": lambda: load_Pheno(args.pheno_file),
        "load-iid": lambda: load_IID(args.snp_file),
        "three-load-iid": lambda: three_load_IID(args.snp_file),
        "load-pheno-id": lambda: load_Pheno_ID(args.pheno_file),
    }

    result = command_map[args.command]()

    # 打印返回结果的摘要
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            if isinstance(item, pd.DataFrame):
                print(f"\n── 返回值 [{i}] DataFrame shape: {item.shape} ──")
            else:
                print(f"\n── 返回值 [{i}]: {type(item).__name__} ──")
    elif isinstance(result, pd.DataFrame):
        print(f"\n── 返回值 DataFrame shape: {result.shape} ──")


if __name__ == "__main__":
    main()
