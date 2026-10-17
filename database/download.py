"""
database/download.py
=====================
高并发并行下载器：通过代理从 HuggingFace 拉取 AMASS 动捕序列及 SMPL-H 体型参数至 ./database。

目录结构:
  database/
    amass/HumanML3D_SMPLH52/{0000..0022, M000..M014}/*.npz   <- 动捕姿态序列
    smplh/{male,female,neutral}/model.npz                    <- SMPL-H 几何模型
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

BASE_DIR = Path(__file__).resolve().parent

# CMU 日常姿态 0000-0022 (23 个子集) + HDM05 技巧姿态 M000-M014 (15 个子集)
CMU_IDS = [f"{i:04d}" for i in range(23)]
HDM05_IDS = [f"M{i:03d}" for i in range(15)]
MOTION_SUBSETS = CMU_IDS + HDM05_IDS

HF_DATASET = "ZeyuLing/MotionHub"
HF_SUBPATH = "HumanML3D_AMASS/smplh_52"

HF_MODEL = "lithiumice/models_hub"
SMPLH_GENDERS = ["male", "female", "neutral"]
SMPLH_SUBPATH = "4_SMPLhub/smplh"


def make_opener(proxy: str | None):
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        h = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(h)


def api_list(opener, kind: str, repo: str, path: str) -> list:
    endpoint = "datasets" if kind == "dataset" else "models"
    url = f"https://huggingface.co/api/{endpoint}/{repo}/tree/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_resolve(kind: str, repo: str, path: str) -> str:
    if kind == "dataset":
        return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def download_single(url: str, dest: Path, proxy: str | None, retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return ("SKIPPED", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = make_opener(proxy)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=40) as r, open(dest, "wb") as f:
                f.write(r.read())
            return ("OK", dest.name)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)
            else:
                return ("ERROR", f"{dest.name}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proxy", default="http://127.0.0.1:7892", help="HTTP 代理地址")
    p.add_argument("--no-proxy", action="store_true", help="禁用代理")
    p.add_argument("--threads", type=int, default=16, help="并发下载线程数")
    p.add_argument("--smplh-only", action="store_true", help="仅下载 SMPL-H 模型")
    p.add_argument("--amass-only", action="store_true", help="仅下载 AMASS 动捕序列")
    args = p.parse_args()

    proxy = None if args.no_proxy else args.proxy
    opener = make_opener(proxy)
    print(f"[*] 代理节点: {proxy or '直连'}")
    print(f"[*] 并发线程数: {args.threads}")

    # 1. 下载 SMPL-H 体型参数
    if not args.amass_only:
        print("\n=== 正在检查与下载 SMPL-H 人体骨骼模型参数 ===")
        for gender in SMPLH_GENDERS:
            dest = BASE_DIR / "smplh" / gender / "model.npz"
            url = hf_resolve("model", HF_MODEL, f"{SMPLH_SUBPATH}/{gender}/model.npz")
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  [√] smplh/{gender}/model.npz 已存在 ({dest.stat().st_size // 1024 // 1024} MB)")
            else:
                print(f"  [+] 正在下载 smplh/{gender}/model.npz ...")
                status, msg = download_single(url, dest, proxy)
                print(f"  [{status}] {gender}: {msg}")

    # 2. 下载 AMASS 序列
    if not args.smplh_only:
        print(f"\n=== 正在收集 AMASS 待下载清单 ({len(MOTION_SUBSETS)} 个子集) ===")
        all_tasks = []
        for subset_id in MOTION_SUBSETS:
            subfolder = f"{HF_SUBPATH}/{subset_id}"
            try:
                entries = api_list(opener, "dataset", HF_DATASET, subfolder)
                npz_files = [e for e in entries if e.get("path", "").endswith(".npz")]
                for entry in npz_files:
                    rel = entry["path"]
                    fname = Path(rel).name
                    dest = BASE_DIR / "amass" / "HumanML3D_SMPLH52" / subset_id / fname
                    url = hf_resolve("dataset", HF_DATASET, rel)
                    all_tasks.append((url, dest))
                print(f"  子集 {subset_id}: 索引到 {len(npz_files)} 个动作片段", flush=True)
            except Exception as e:
                print(f"  [!] 索引子集 {subset_id} 失败: {e}", flush=True)

        total_count = len(all_tasks)
        print(f"\n[*] 任务队列就绪: 共计 {total_count} 个 .npz 动捕文件待同步，启动多线程下载...")
        done_count = 0
        ok_count = 0
        skip_count = 0
        err_count = 0

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {pool.submit(download_single, url, dest, proxy): dest for url, dest in all_tasks}
            start_time = time.time()
            for fut in as_completed(futures):
                done_count += 1
                status, msg = fut.result()
                if status == "OK":
                    ok_count += 1
                elif status == "SKIPPED":
                    skip_count += 1
                else:
                    err_count += 1
                    print(f"\n  [ERROR] {msg}", flush=True)

                if done_count % 25 == 0 or done_count == total_count:
                    elapsed = time.time() - start_time
                    speed = done_count / max(elapsed, 0.001)
                    percent = (done_count / total_count) * 100
                    print(f"\r进度: [{done_count}/{total_count}] ({percent:.1f}%) | 新增: {ok_count} | 跳过: {skip_count} | 失败: {err_count} | 速度: {speed:.1f} 文件/秒", end="", flush=True)

        print("\n\n=== 同步全部完成 ===")
        print(f"总计: {total_count} | 新下载: {ok_count} | 校验跳过: {skip_count} | 失败: {err_count}")


if __name__ == "__main__":
    main()
