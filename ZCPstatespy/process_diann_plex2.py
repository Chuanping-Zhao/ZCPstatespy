# process_diann_plex2.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import csv
from typing import Optional
import numpy as np
import pandas as pd

# =========================
# 配置（可按需修改）
# =========================
RAW_PATH = "report.tsv"                          # DIA-NN 原始 report
OUT_PATH = "report_plex2_s0s8.tsv"               # 输出路径
CHANNEL_QVAL_CUTOFF = 0.15                       # Channel.Q.Value 阈值
# 例：自动截到 TEST_Rx 前的固定段，可用正则 r".*?_200ng_"
RUN_PREFIX: Optional[str] = None                 # 若不想截，设为 None
RUN_PREFIX_IS_REGEX: bool = False                # 若 RUN_PREFIX 是正则，设为 True

# =========================
# 工具函数
# =========================
def assign_channel_infor(mod_seq: str) -> str:
    """根据 Modified.Sequence 判定 s0/s4/s8/none"""
    if pd.isna(mod_seq):
        return "none"
    if re.match(r"^\(Dimethyl-n-0\)", mod_seq) or re.match(r"^\(Dimethyl\)", mod_seq):
        return "s0"
    if re.match(r"^\(Dimethyl-n-4\)", mod_seq):
        return "s4"
    if re.match(r"^\(Dimethyl-n-8\)", mod_seq):
        return "s8"
    return "none"


def replace_to_s4(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"\(Dimethyl-n-8\)", "(Dimethyl-n-4)", x)
    x = re.sub(r"\(Dimethyl-K-8\)", "(Dimethyl-K-4)", x)
    return x


def replace_to_s8(x: str) -> str:
    if pd.isna(x):
        return x
    x = re.sub(r"\(Dimethyl-n-4\)", "(Dimethyl-n-8)", x)
    x = re.sub(r"\(Dimethyl-K-4\)", "(Dimethyl-K-8)", x)
    return x


def reorder_channel4_after_channel0(df: pd.DataFrame) -> pd.DataFrame:
    """把 Channel.4 放到 Channel.0 后面（若两者都存在）"""
    cols = list(df.columns)
    if "Channel.0" in cols and "Channel.4" in cols:
        cols.remove("Channel.4")
        idx0 = cols.index("Channel.0")
        cols.insert(idx0 + 1, "Channel.4")
        df = df.loc[:, cols]
    return df


def _cut_by_prefix(name: str, run_prefix: str, is_regex: bool) -> str:
    """根据前缀（字面量或正则）裁剪 name 前缀；失败返回原 name"""
    if not run_prefix:
        return name
    if is_regex:
        m = re.search(run_prefix, name)
        return name[m.end():] if m else name
    # 字面量：用正向后顾
    m = re.search(rf"(?<={re.escape(run_prefix)}).*", name)
    return m.group(0) if m else name


def mutate_run(run_val: str,
               channel: str,
               run_prefix: Optional[str] = None,
               run_prefix_is_regex: bool = False) -> str:
    """
    重写 Run：
    - 先 basename 去路径，只保留文件名
    - 若提供 run_prefix：
        * is_regex=False -> 作为字面量前缀裁剪
        * is_regex=True  -> 作为正则裁剪（取匹配末尾后的子串）
      否则不截
    - 拼接 "_{channel}"
    - 裁掉 R# 后面的尾巴（保留 TEST_R1/2/3）
    """
    if pd.isna(run_val):
        return run_val
    base = os.path.basename(str(run_val))
    extracted = _cut_by_prefix(base, run_prefix, run_prefix_is_regex) if run_prefix is not None else base
    extracted = f"{extracted}_{channel}"
    extracted = re.sub(r"(?<=R\d).*", "", extracted)
    return extracted


def ensure_channel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 Channel.4 和 Channel.8 存在；
    若缺失则互相复制，若都缺失则报错。
    """
    has_c4 = "Channel.4" in df.columns
    has_c8 = "Channel.8" in df.columns

    if not has_c4 and has_c8:
        print("⚠️ 未发现 Channel.4，已用 Channel.8 复制生成。")
        df["Channel.4"] = df["Channel.8"]
    elif not has_c8 and has_c4:
        print("⚠️ 未发现 Channel.8，已用 Channel.4 复制生成。")
        df["Channel.8"] = df["Channel.4"]
    elif not has_c4 and not has_c8:
        raise ValueError("输入表缺少 Channel.4 和 Channel.8，无法补充通道。")
    return df


def fill_dimethyl_channels(df: pd.DataFrame) -> pd.DataFrame:
    """补齐 s4 / s8 通道（对称复制赋值）"""
    presence = (
        df.groupby("Run")
        .agg(
            has0=("channelInfor", lambda s: (s == "s0").any()),
            has4=("channelInfor", lambda s: (s == "s4").any()),
            has8=("channelInfor", lambda s: (s == "s8").any()),
        )
        .reset_index()
    )

    runs_need_s4 = presence.loc[presence["has0"] & presence["has8"] & (~presence["has4"]), "Run"].tolist()
    runs_need_s8 = presence.loc[presence["has0"] & presence["has4"] & (~presence["has8"]), "Run"].tolist()

    # ---- 补 s4（从 s8 复制；Channel.4 = Channel.8）
    add_s4_df = pd.DataFrame()
    if runs_need_s4:
        mask = df["Run"].isin(runs_need_s4) & (df["channelInfor"] == "s8")
        add_s4_df = df.loc[mask].copy()
        if not add_s4_df.empty:
            add_s4_df["channelInfor"] = "s4"
            if "Modified.Sequence" in add_s4_df.columns:
                add_s4_df["Modified.Sequence"] = add_s4_df["Modified.Sequence"].map(replace_to_s4)
            if "Precursor.Id" in add_s4_df.columns:
                add_s4_df["Precursor.Id"] = add_s4_df["Precursor.Id"].map(replace_to_s4)
            if "Channel.8" in add_s4_df.columns:
                add_s4_df["Channel.4"] = add_s4_df["Channel.8"]

    # ---- 补 s8（从 s4 复制；Channel.8 = Channel.4）
    add_s8_df = pd.DataFrame()
    if runs_need_s8:
        mask = df["Run"].isin(runs_need_s8) & (df["channelInfor"] == "s4")
        add_s8_df = df.loc[mask].copy()
        if not add_s8_df.empty:
            add_s8_df["channelInfor"] = "s8"
            if "Modified.Sequence" in add_s8_df.columns:
                add_s8_df["Modified.Sequence"] = add_s8_df["Modified.Sequence"].map(replace_to_s8)
            if "Precursor.Id" in add_s8_df.columns:
                add_s8_df["Precursor.Id"] = add_s8_df["Precursor.Id"].map(replace_to_s8)
            if "Channel.4" in add_s8_df.columns:
                add_s8_df["Channel.8"] = add_s8_df["Channel.4"]

    out = pd.concat([df, add_s4_df, add_s8_df], axis=0, ignore_index=True)
    if {"Run", "channelInfor"}.issubset(out.columns):
        out.sort_values(by=["Run", "channelInfor"], inplace=True, kind="mergesort")
    return out
def read_input_file(input_path: str) -> pd.DataFrame:
    """支持 .tsv 和 .parquet 输入"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(input_path, sep="\t", encoding="latin1")
    elif ext == ".parquet":
        return pd.read_parquet(input_path)  # 依赖 pyarrow 或 fastparquet
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


def process_diann_and_fill_channels(
    input_path: str,
    output_path: str,
    channel_q_cutoff: float = 0.15,
    run_prefix: Optional[str] = RUN_PREFIX,
    run_prefix_is_regex: bool = RUN_PREFIX_IS_REGEX,
) -> pd.DataFrame:
    """
    完整流程：
    读 DIA-NN → 判定通道 → 过滤 → Run/File.Name → 检查/补齐 → 补通道 → 导出
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = read_input_file(input_path)

    # 判定通道标签
    df["channelInfor"] = df["Modified.Sequence"].apply(assign_channel_infor)

    # 检查并补齐 Channel.4 / Channel.8
    df = ensure_channel_columns(df)

    # 过滤 Channel.Q.Value
    if "Channel.Q.Value" not in df.columns:
        raise ValueError("输入表缺少列：'Channel.Q.Value'。")
    df = df[df["Channel.Q.Value"] <= channel_q_cutoff].copy()

    # 重写 Run 与 File.Name（先 basename，再按需裁剪前缀）
    df["Run"] = [
        mutate_run(r, c, run_prefix=run_prefix, run_prefix_is_regex=run_prefix_is_regex)
        for r, c in zip(df["Run"], df["channelInfor"])
    ]
    df["File.Name"] = df["Run"]

    # 按 Run 补通道
    df = fill_dimethyl_channels(df)

    # 列顺序：把 Channel.4 放到 Channel.0 后
    df = reorder_channel4_after_channel0(df)

    # 删除 channelInfor
    if "channelInfor" in df.columns:
        df.drop(columns=["channelInfor"], inplace=True)

    # 导出（兼容不同 pandas 版本）
    try:
        df.to_csv(
            output_path,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_NONE,
            na_rep="",
            line_terminator="\n",   # 新版 pandas
            encoding="utf-8-sig"
        )
    except TypeError:
        df.to_csv(
            output_path,
            sep="\t",
            index=False,
            quoting=csv.QUOTE_NONE,
            na_rep="",
            lineterminator="\n",    # 旧版 pandas
            encoding="utf-8-sig"
        )

    print(f"✅ Done. Wrote: {output_path}")
    if "Run" in df.columns:
        print("Runs:", ", ".join(sorted(df["Run"].unique())))

    return df


# =========================
# 命令行入口
# =========================
if __name__ == "__main__":
    in_path = RAW_PATH
    out_path = OUT_PATH
    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    # 例：不同日期 + 相同 200ng 前缀时，用正则更稳
    process_diann_and_fill_channels(
        input_path=in_path,
        output_path=out_path,
        channel_q_cutoff=CHANNEL_QVAL_CUTOFF,
        run_prefix=RUN_PREFIX,            # None 表示不截
        run_prefix_is_regex=RUN_PREFIX_IS_REGEX
    )
