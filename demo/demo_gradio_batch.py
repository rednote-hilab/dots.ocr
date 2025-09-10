import os
import io
import uuid
import json
import zipfile
import tempfile
import threading
import queue
import shutil
from pathlib import Path
from PIL import Image
import requests
import gradio as gr
import re
import math
import datetime

# Local project imports (assumed available)
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.demo_utils.display import read_image
from dots_ocr.parser import DotsOCRParser

# ---------------- Config & globals ----------------
DEFAULT_CONFIG = {
    "ip": "127.0.0.1",
    "port_vllm": 8000,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
}

# Absolute constraints discovered from runtime:
ABS_MIN_PIXELS = 3136
ABS_MAX_PIXELS = 11289600

current_config = DEFAULT_CONFIG.copy()

# default parser instance (can be overridden per-task)
dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG["ip"],
    port=DEFAULT_CONFIG["port_vllm"],
    dpi=200,
    min_pixels=DEFAULT_CONFIG["min_pixels"],
    max_pixels=DEFAULT_CONFIG["max_pixels"],
)

RESULTS_CACHE = {}  # rid -> result dict or placeholder
TASK_QUEUE = queue.Queue()
# Worker pool for background processing (adjustable via UI)
WORKER_THREADS = []
MAX_CONCURRENCY = 6
THREAD_LOCK = threading.Lock()
RETRY_COUNTS = {}  # rid -> attempts
MAX_AUTO_RETRIES = 5
RETRY_BACKOFF_BASE = 1.7
DEFAULT_SCRIPT_TEMPLATE = """# 高级脚本使用说明
# 提供对象: api
# 日志: 使用 print(...) 或 debug(...) 输出到下方“脚本日志”实时区域。
# api.get_ids() -> [rid,...] 按当前 UI 顺序返回
# api.get_status(rid) -> {'status','ui': {'tab','nohf','source'}, 'filtered': bool, 'input_width': int, 'input_height': int}
# api.get_texts(rid) -> {
#   'md': 原始 Markdown, 'md_nohf': 原始 NOHF Markdown, 'json': 原始 JSON,
#   'md_edit': 编辑版 Markdown 或 None, 'md_nohf_edit': 编辑版 NOHF Markdown 或 None, 'json_edit': 编辑版 JSON 或 None
# }
# api.choose_texts(rid, prefer_ui=True, prefer_edit=True, prefer_nohf=None) -> {'md','json'}
#   - prefer_ui: True 时根据当前 UI 的 NOHF/来源选择内容
#   - prefer_edit: True 时优先用编辑内容（若存在）
#   - prefer_nohf: 显式指定是否使用 NOHF（覆盖 UI），None 表示跟随 UI
# api.list_paths(rid) -> {
#   'temp_dir': str, 'session_id': str,
#   'result': {'md':path,'md_nohf':path,'json':path,'layout':path or None,'image':path or None},
#   'edited': {'md':path or None,'md_nohf':path or None,'json':path or None}
# }
# api.path_exists(path) -> bool   判断路径是否存在
# api.build_export(name='custom') -> ExportBuilder
# ExportBuilder:
#   .add_text('dir/file.md', '...')            写入文本
#   .add_bytes('bin/data.bin', b'...')         写入二进制
#   .add_file('/abs/path/file.md', 'dir/file.md')  拷贝已有文件
#   .mkdir('subdir/')                           创建目录
#   .finalize() -> zip_path                     打包为 zip 并返回路径
#
# 约定: 定义 main(api) 并返回以下之一：
# - ExportBuilder 实例（将自动 finalize）
# - 目录路径或文件路径（目录将被打包为 zip）
# - None（若存在变量 export=ExportBuilder，将自动 finalize）
#
# 示例：按 UI 所见优先使用“编辑源码”与 NOHF，导出每个结果的 md/json，同时附带原始与编辑文件
def main(api):
    ids = api.get_ids()
    eb = api.build_export('custom_export')
    for i, rid in enumerate(ids, start=1):
        st = api.get_status(rid)
        if st['status'] != 'done':
            continue
        choice = api.choose_texts(rid, prefer_ui=True, prefer_edit=True)
        eb.add_text(f'result_{i}_{rid}/content.md', choice['md'] or '')
        eb.add_text(f'result_{i}_{rid}/data.json', choice['json'] or '{}')
        paths = api.list_paths(rid)
        # 附带原始文件
        for p in (paths.get('result') or {}).values():
            if p and api.path_exists(p):
                name = Path(p).name
                eb.add_file(p, f'result_{i}_{rid}/raw/{name}')
        # 附带编辑文件
        for p in (paths.get('edited') or {}).values():
            if p and api.path_exists(p):
                name = Path(p).name
                eb.add_file(p, f'result_{i}_{rid}/edited/{name}')
    return eb
"""


# ---------------- Helpers ----------------
def read_image_v2(img):
    """Read image from URL or local path / PIL.Image. Supports file paths and URLs."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str) and img.startswith(("http://", "https://")):
        with requests.get(img, stream=True) as r:
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    if isinstance(img, str) and os.path.exists(img):
        return Image.open(img).convert("RGB")
    try:
        img_res = read_image(img, use_native=True)
        if isinstance(img_res, tuple) and isinstance(img_res[0], Image.Image):
            return img_res[0]
    except Exception:
        pass
    raise ValueError(f"Unsupported image input: {type(img)} / {repr(img)[:200]}")


def create_temp_session_dir():
    session_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_demo_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir, session_id


def classify_parse_failure(exc, min_p, max_p):
    """Return a user-friendly error message for known failure causes."""
    msg = str(exc)
    reasons = []
    # Absolute & semantic constraints
    if min_p < ABS_MIN_PIXELS:
        reasons.append(
            f"Min Pixels 过小：{min_p}，必须 >= {ABS_MIN_PIXELS}。建议提高 Min Pixels。"
        )
    if max_p > ABS_MAX_PIXELS:
        reasons.append(
            f"Max Pixels 过大：{max_p}，必须 <= {ABS_MAX_PIXELS}。建议降低 Max Pixels。"
        )
    if min_p >= max_p:
        reasons.append(
            f"像素参数不合法：Min Pixels({min_p}) >= Max Pixels({max_p})，必须满足 Min Pixels < Max Pixels。"
        )

    lower = msg.lower()
    if "no results returned from parser" in lower or "no results returned" in lower:
        reasons.append(
            "解析未返回结果。可能原因：图像过小、Min Pixels 设置过小或过滤过强。"
            f"建议：Min Pixels >= {ABS_MIN_PIXELS} 且 Max Pixels <= {ABS_MAX_PIXELS}。"
        )
    if "failed to read input" in lower or "cannot identify image file" in lower:
        reasons.append("无法读取输入文件，请确认文件是否为有效图片或PDF。")
    if ("connection" in lower and "refused" in lower) or ("connectionerror" in lower):
        reasons.append("无法连接后端推理服务，请检查 Server IP/Port 与服务状态。")

    if not reasons:
        reasons.append(f"未知错误：{msg}")

    detail = "\n".join(f"- {r}" for r in reasons)
    cfg = f"(当前参数：min_pixels={min_p}, max_pixels={max_p})"
    return f"解析失败：\n{detail}\n{cfg}"


def _is_transient_backend_error(exc: Exception):
    lower = str(exc).lower()
    # Common signals: connection refused/reset, timeout, gateway, service unavailable
    keywords = [
        "connection refused",
        "connectionerror",
        "timeout",
        "timed out",
        "gateway",
        "service unavailable",
        "failed to establish a new connection",
        "max retries exceeded",
        "read timeout",
        "connect timeout",
    ]
    return any(k in lower for k in keywords)


def parse_image_with_high_level_api(parser, image, prompt_mode, fitz_preprocess=False):
    """
    Calls parser.parse_image with a PIL image (or accepts image path if parser expects path).
    Returns dictionary with artifacts. Keeps a temp PNG of the input for traceability.
    """
    temp_dir, session_id = create_temp_session_dir()
    if not isinstance(image, Image.Image):
        image = read_image_v2(image)
    temp_image_path = os.path.join(temp_dir, f"input_{session_id}.png")
    image.save(temp_image_path, "PNG")

    filename = f"demo_{session_id}"
    results = parser.parse_image(
        input_path=image,
        filename=filename,
        prompt_mode=prompt_mode,
        save_dir=temp_dir,
        fitz_preprocess=fitz_preprocess,
    )
    if not results:
        raise RuntimeError("No results returned from parser")

    result = results[0]
    layout_image = None
    if result.get("layout_image_path") and os.path.exists(result["layout_image_path"]):
        try:
            layout_image = Image.open(result["layout_image_path"]).convert("RGB")
        except Exception:
            layout_image = None

    cells_data = None
    if result.get("layout_info_path") and os.path.exists(result["layout_info_path"]):
        with open(result["layout_info_path"], "r", encoding="utf-8") as f:
            cells_data = json.load(f)

    md_content = None
    if result.get("md_content_path") and os.path.exists(result["md_content_path"]):
        with open(result["md_content_path"], "r", encoding="utf-8") as f:
            md_content = f.read()

    md_content_nohf = None
    if result.get("md_content_nohf_path") and os.path.exists(
        result["md_content_nohf_path"]
    ):
        with open(result["md_content_nohf_path"], "r", encoding="utf-8") as f:
            md_content_nohf = f.read()

    json_code = ""
    if cells_data is not None:
        try:
            json_code = json.dumps(cells_data, ensure_ascii=False, indent=2)
        except Exception:
            json_code = str(cells_data)

    return {
        "original_image": image,
        "layout_image": layout_image,
        "cells_data": cells_data,
        "md_content": md_content,
        "md_content_nohf": md_content_nohf,
        "json_code": json_code,
        "filtered": result.get("filtered", False),
        "temp_dir": temp_dir,
        "session_id": session_id,
        "result_paths": result,
        "input_width": result.get("input_width", 0),
        "input_height": result.get("input_height", 0),
        "input_temp_path": temp_image_path,
    }


def _validate_pixels(min_p, max_p):
    """Coerce pixel parameters. Do NOT auto-swap; semantic errors are handled by pre-validation."""
    try:
        min_p = int(min_p)
    except Exception:
        min_p = DEFAULT_CONFIG["min_pixels"]
    try:
        max_p = int(max_p)
    except Exception:
        max_p = DEFAULT_CONFIG["max_pixels"]
    if min_p <= 0:
        min_p = DEFAULT_CONFIG["min_pixels"]
    if max_p <= 0:
        max_p = DEFAULT_CONFIG["max_pixels"]
    return min_p, max_p


def _set_parser_config(server_ip, server_port, min_pixels, max_pixels):
    min_pixels, max_pixels = _validate_pixels(min_pixels, max_pixels)
    current_config.update(
        {
            "ip": server_ip,
            "port_vllm": int(server_port),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
    )
    dots_parser.ip = server_ip
    dots_parser.port = int(server_port)
    dots_parser.min_pixels = min_pixels
    dots_parser.max_pixels = max_pixels


def purge_queue(rid):
    """Best-effort remove tasks matching rid from queue."""
    pending = []
    try:
        while True:
            task = TASK_QUEUE.get_nowait()
            if task and isinstance(task, tuple):
                if task[0] != rid:
                    pending.append(task)
            TASK_QUEUE.task_done()
    except queue.Empty:
        pass
    for t in pending:
        TASK_QUEUE.put(t)


# ---------------- Export helpers ----------------
def export_one_rid(rid):
    st = RESULTS_CACHE.get(rid)
    if not st:
        return None
    temp_dir = st.get("temp_dir")
    if not temp_dir or not os.path.isdir(temp_dir):
        return None
    out_dir, _sess = create_temp_session_dir()
    zip_path = os.path.join(out_dir, f"export_{rid}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rt, _, files in os.walk(temp_dir):
            for f in files:
                src = os.path.join(rt, f)
                rel = os.path.relpath(src, temp_dir)
                zf.write(src, os.path.join(f"result_{rid}", rel))
    return zip_path


def ensure_export_ready(rid):
    """Create and cache export zip path if not present."""
    st = RESULTS_CACHE.get(rid) or {}
    if not st or st.get("status") != "done":
        return None
    path = st.get("export_path")
    if path and os.path.exists(path):
        return path
    path = export_one_rid(rid)
    if path:
        st["export_path"] = path
        RESULTS_CACHE[rid] = st
    return path


# ---------------- Script API & execution ----------------
class ExportBuilder:
    def __init__(self, name=None):
        root, sid = create_temp_session_dir()
        sub = f"script_export_{sid}"
        if name:
            sub = f"{name}_{sid}"
        self.root_dir = os.path.join(root, sub)
        os.makedirs(self.root_dir, exist_ok=True)
        self._final_zip = None

    def _abspath(self, rel_path: str):
        rel_path = rel_path.lstrip("/\\")
        return os.path.join(self.root_dir, rel_path)

    def mkdir(self, rel_dir: str):
        p = self._abspath(rel_dir)
        os.makedirs(p, exist_ok=True)
        return p

    def add_text(self, rel_path: str, content: str, encoding: str = "utf-8"):
        p = self._abspath(rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding=encoding) as f:
            f.write("" if content is None else str(content))
        return p

    def add_bytes(self, rel_path: str, data: bytes):
        p = self._abspath(rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data or b"")
        return p

    def add_file(self, src_path: str, dest_rel_path: str = None):
        if not src_path or not os.path.exists(src_path):
            return None
        dest_rel_path = dest_rel_path or os.path.basename(src_path)
        p = self._abspath(dest_rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy2(src_path, p)
        return p

    def finalize(self, zip_name: str = None):
        if self._final_zip and os.path.exists(self._final_zip):
            return self._final_zip
        out_dir, sid = create_temp_session_dir()
        zip_name = zip_name or f"script_export_{sid}.zip"
        zip_path = os.path.join(out_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rt, _, files in os.walk(self.root_dir):
                for f in files:
                    src = os.path.join(rt, f)
                    rel = os.path.relpath(src, self.root_dir)
                    zf.write(src, rel)
        self._final_zip = zip_path
        return zip_path


class ScriptAPI:
    def __init__(self, ids_snapshot):
        self._ids = list(ids_snapshot or [])

    def get_ids(self):
        return list(self._ids)

    def get_status(self, rid: str):
        st = dict(RESULTS_CACHE.get(rid) or {})
        ui = dict(st.get("ui") or {})
        return {
            "status": st.get("status", "pending"),
            "ui": {
                "tab": ui.get("tab", "md"),
                "nohf": bool(ui.get("nohf", False)),
                "source": ui.get("source", "源码"),
            },
            "filtered": bool(st.get("filtered", False)),
            "input_width": int(st.get("input_width", 0) or 0),
            "input_height": int(st.get("input_height", 0) or 0),
        }

    def get_texts(self, rid: str):
        st = dict(RESULTS_CACHE.get(rid) or {})
        edits = dict(st.get("edits") or {})
        return {
            "md": st.get("md_content") or "",
            "md_nohf": st.get("md_content_nohf") or "",
            "json": st.get("json_code") or "",
            "md_edit": edits.get("md"),
            "md_nohf_edit": edits.get("nohf"),
            "json_edit": edits.get("json"),
        }

    def choose_texts(
        self,
        rid: str,
        prefer_ui: bool = True,
        prefer_edit: bool = True,
        prefer_nohf: bool | None = None,
    ):
        st = dict(RESULTS_CACHE.get(rid) or {})
        ui = dict(st.get("ui") or {})
        # UI 指示
        ui_nohf = bool(ui.get("nohf", False))
        ui_source_is_edit = str(ui.get("source", "源码")) == "编辑源码"
        # 选择 nohf
        use_nohf = ui_nohf if prefer_nohf is None else bool(prefer_nohf)
        # 选择是否优先编辑
        prefer_edit_final = bool(prefer_edit or (prefer_ui and ui_source_is_edit))
        t = self.get_texts(rid)
        # Markdown
        md_orig = t["md_nohf"] if use_nohf else t["md"]
        md_edit = t["md_nohf_edit"] if use_nohf else t["md_edit"]
        md = (md_edit if (prefer_edit_final and md_edit is not None) else md_orig) or ""
        # JSON
        json_text = (
            t["json_edit"]
            if (prefer_edit_final and t.get("json_edit") is not None)
            else t["json"]
        ) or ""
        return {"md": md, "json": json_text}

    def list_paths(self, rid: str):
        st = dict(RESULTS_CACHE.get(rid) or {})
        rp = dict(st.get("result_paths") or {})
        md_p = rp.get("md_content_path")
        nohf_p = rp.get("md_content_nohf_path")
        json_p = rp.get("layout_info_path") or rp.get("json_path")
        image_p = rp.get("layout_image_path") or None
        # 编辑路径（若存在）
        edited_md = None
        edited_nohf = None
        edited_json = None
        try:
            edited_md = _edited_filepath(st, "md")
            if not os.path.exists(edited_md):
                edited_md = None
        except Exception:
            edited_md = None
        try:
            edited_nohf = _edited_filepath(st, "nohf")
            if not os.path.exists(edited_nohf):
                edited_nohf = None
        except Exception:
            edited_nohf = None
        try:
            edited_json = _edited_filepath(st, "json")
            if not os.path.exists(edited_json):
                edited_json = None
        except Exception:
            edited_json = None
        return {
            "temp_dir": st.get("temp_dir"),
            "session_id": st.get("session_id"),
            "result": {
                "md": md_p if (md_p and os.path.exists(md_p)) else None,
                "md_nohf": nohf_p if (nohf_p and os.path.exists(nohf_p)) else None,
                "json": json_p if (json_p and os.path.exists(json_p)) else None,
                "layout": image_p if (image_p and os.path.exists(image_p)) else None,
                "input_image": (
                    st.get("input_temp_path")
                    if (
                        st.get("input_temp_path")
                        and os.path.exists(st.get("input_temp_path"))
                    )
                    else None
                ),
            },
            "edited": {
                "md": edited_md,
                "md_nohf": edited_nohf,
                "json": edited_json,
            },
        }

    def path_exists(self, p: str) -> bool:
        try:
            return bool(p) and os.path.exists(p)
        except Exception:
            return False

    def build_export(self, name: str | None = None):
        return ExportBuilder(name=name)


def _safe_builtins():
    base = (
        __builtins__
        if isinstance(__builtins__, dict)
        else getattr(__builtins__, "__dict__", {})
    )
    allow = [
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "enumerate",
        "map",
        "filter",
        "zip",
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "print",
        "any",
        "all",
        "sorted",
    ]
    return {k: base[k] for k in allow if k in base}


def run_user_script(script_code: str, ids_snapshot):
    """
    非流式执行用户脚本，捕获标准输出并返回（zip_path, logs）。
    """
    api = ScriptAPI(ids_snapshot)
    ns = {
        "__builtins__": _safe_builtins(),
        "api": api,
        "json": json,
        "re": re,
        "math": math,
        "datetime": datetime,
        "Path": Path,
        "io": io,
        "ExportBuilder": ExportBuilder,
    }
    import contextlib
    from io import StringIO

    buf = StringIO()
    zip_path = None
    try:
        code = script_code or ""
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
            result = None
            main_fn = ns.get("main")
            if callable(main_fn):
                result = main_fn(api)
            else:
                result = ns.get("RESULT") or ns.get("OUTPUT_PATH")
            if isinstance(result, ExportBuilder):
                zip_path = result.finalize()
            elif isinstance(result, str) and result:
                if os.path.isdir(result):
                    eb = ExportBuilder("script_dir_export")
                    for rt, _, files in os.walk(result):
                        for f in files:
                            src = os.path.join(rt, f)
                            rel = os.path.relpath(src, result)
                            eb.add_file(src, rel)
                    zip_path = eb.finalize()
                elif os.path.exists(result):
                    zip_path = result
            if not zip_path:
                exp = ns.get("export")
                if isinstance(exp, ExportBuilder):
                    zip_path = exp.finalize()
    except Exception as e:
        err = f"[Script Error] {type(e).__name__}: {e}"
        return None, (buf.getvalue() + "\n" + err)
    return (
        zip_path if (zip_path and os.path.exists(zip_path)) else None
    ), buf.getvalue()


def run_user_script_stream(script_code: str, ids_snapshot):
    """生成器：实时输出日志，并在结束时返回下载地址与完成状态。"""
    # 日志队列
    log_q = queue.Queue()

    def _emit(kind, payload=None):
        log_q.put((kind, payload))

    def debug(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text:
            _emit("log", text)

    # 准备脚本命名空间（与非流式版本一致，但覆盖 print/debug）
    api = ScriptAPI(ids_snapshot)
    ns = {
        "__builtins__": _safe_builtins(),
        "api": api,
        "json": json,
        "re": re,
        "math": math,
        "datetime": datetime,
        "Path": Path,
        "io": io,
        "ExportBuilder": ExportBuilder,
        # 专用日志函数
        "debug": debug,
        "print": debug,
    }

    result_holder = {"zip_path": None, "error": None}

    def _worker():
        try:
            code = script_code or ""
            exec(code, ns, ns)
            res = None
            main_fn = ns.get("main")
            if callable(main_fn):
                res = main_fn(api)
            else:
                res = ns.get("RESULT") or ns.get("OUTPUT_PATH")
            zip_path = None
            if isinstance(res, ExportBuilder):
                zip_path = res.finalize()
            elif isinstance(res, str) and res:
                if os.path.isdir(res):
                    eb = ExportBuilder("script_dir_export")
                    for rt, _, files in os.walk(res):
                        for f in files:
                            src = os.path.join(rt, f)
                            rel = os.path.relpath(src, res)
                            eb.add_file(src, rel)
                    zip_path = eb.finalize()
                elif os.path.exists(res):
                    zip_path = res
            if not zip_path:
                exp = ns.get("export")
                if isinstance(exp, ExportBuilder):
                    zip_path = exp.finalize()
            result_holder["zip_path"] = (
                zip_path if (zip_path and os.path.exists(zip_path)) else None
            )
        except Exception as e:
            result_holder["error"] = f"[Script Error] {type(e).__name__}: {e}"
        finally:
            _emit("done", None)

    # 启动脚本线程
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # 初始状态显示
    spinner_html = (
        "<div style='display:flex;align-items:center;gap:8px;'>"
        "<svg width='18' height='18' viewBox='0 0 50 50' style='animation:spin 1s linear infinite'>"
        "<circle cx='25' cy='25' r='20' stroke='#FF576D' stroke-width='4' fill='none' stroke-linecap='round' "
        "stroke-dasharray='31.4 31.4'>"  # dash pattern for arc
        "</circle></svg>"
        "<style>@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}</style>"
        "<span>脚本运行中…</span></div>"
    )
    log_buf_lines = []
    # 初始仅显示运行中动画，日志区域留空
    yield None, spinner_html, ""

    # 实时拉取日志并渲染
    while True:
        try:
            kind, payload = log_q.get(timeout=0.2)
        except queue.Empty:
            if not t.is_alive():
                # 线程已结束但没有新的事件，跳到收尾
                break
            else:
                continue

        if kind == "log":
            # 追加日志并推送更新
            if isinstance(payload, str):
                for line in payload.splitlines() or [payload]:
                    if line.strip() == "":
                        continue
                    log_buf_lines.append(line)
            yield None, spinner_html, "```\n" + "\n".join(
                log_buf_lines[-200:]
            ) + "\n```"  # 限制最后200行
        elif kind == "done":
            break

    # 收尾：根据结果/错误输出最终状态
    if result_holder.get("error"):
        log_buf_lines.append(result_holder["error"])
        status_html = (
            "<div style='display:flex;align-items:center;gap:8px;color:#fca5a5'>"
            "<span>❌ 脚本执行失败</span></div>"
        )
        yield None, status_html, "```\n" + "\n".join(log_buf_lines[-500:]) + "\n```"
    else:
        status_html = (
            "<div style='display:flex;align-items:center;gap:8px;color:#86efac'>"
            "<span>✅ 脚本执行完成</span></div>"
        )
        if result_holder.get("zip_path"):
            yield result_holder["zip_path"], status_html, "```\n" + "\n".join(
                log_buf_lines[-500:]
            ) + "\n```"
        else:
            log_buf_lines.append(
                "(无可下载文件返回，若需导出请返回 ExportBuilder 或目录/文件路径)"
            )
            yield None, status_html, "```\n" + "\n".join(log_buf_lines[-500:]) + "\n```"
    """
    执行用户脚本，返回 (zip_path or None, log_text)
    """
    api = ScriptAPI(ids_snapshot)
    ns = {
        "__builtins__": _safe_builtins(),
        "api": api,
        # 常用库（只读注入）
        "json": json,
        "re": re,
        "math": math,
        "datetime": datetime,
        "Path": Path,
        "io": io,
        # 导出构建器类型（如需构造）
        "ExportBuilder": ExportBuilder,
    }
    import contextlib
    from io import StringIO

    buf = StringIO()
    zip_path = None
    try:
        code = script_code or ""
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
            result = None
            main_fn = ns.get("main")
            if callable(main_fn):
                result = main_fn(api)
            else:
                result = ns.get("RESULT") or ns.get("OUTPUT_PATH")
            # 结果归档处理
            if isinstance(result, ExportBuilder):
                zip_path = result.finalize()
            elif isinstance(result, str) and result:
                if os.path.isdir(result):
                    eb = ExportBuilder("script_dir_export")
                    for rt, _, files in os.walk(result):
                        for f in files:
                            src = os.path.join(rt, f)
                            rel = os.path.relpath(src, result)
                            eb.add_file(src, rel)
                    zip_path = eb.finalize()
                elif os.path.exists(result):
                    zip_path = result
            if not zip_path:
                exp = ns.get("export")
                if isinstance(exp, ExportBuilder):
                    zip_path = exp.finalize()
    except Exception as e:
        err = f"[Script Error] {type(e).__name__}: {e}"
        return None, (buf.getvalue() + "\n" + err)
    return (
        zip_path if (zip_path and os.path.exists(zip_path)) else None
    ), buf.getvalue()


def export_selected_rids(ids, selected_labels):
    """
    Build a combined zip for multiple selected results based on their current images (no reupload).
    Only includes items with status == 'done'.
    """
    if not ids or not selected_labels:
        return None
    # Map labels "Result N" -> indices
    sel_indices = []
    for label in selected_labels:
        try:
            idx = int(str(label).split()[-1]) - 1
            if 0 <= idx < len(ids):
                sel_indices.append(idx)
        except Exception:
            continue
    if not sel_indices:
        return None

    out_dir, session_id = create_temp_session_dir()
    zip_path = os.path.join(out_dir, f"export_selected_{session_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in sel_indices:
            rid = ids[i]
            st = RESULTS_CACHE.get(rid) or {}
            if st.get("status") != "done":
                continue
            temp_dir = st.get("temp_dir")
            if not temp_dir or not os.path.isdir(temp_dir):
                # fallback: ensure individual export then include that zip
                single_zip = ensure_export_ready(rid)
                if single_zip and os.path.exists(single_zip):
                    zf.write(single_zip, os.path.join(f"result_{i+1}_{rid}.zip"))
                continue
            base_dir = f"result_{i+1}_{rid}"
            for rt, _, files in os.walk(temp_dir):
                for f in files:
                    src = os.path.join(rt, f)
                    rel = os.path.relpath(src, temp_dir)
                    zf.write(src, os.path.join(base_dir, rel))
    return zip_path if os.path.exists(zip_path) else None


# --------- Edited sources helpers ----------
def _get_base_name_from_result(st: dict):
    """Infer base filename like 'demo_xxx' from result paths or session id."""
    rp = st.get("result_paths") or {}
    for key in ("md_content_path", "md_content_nohf_path", "layout_info_path"):
        p = rp.get(key)
        if p and isinstance(p, str):
            base = os.path.splitext(os.path.basename(p))[0]
            if key == "md_content_nohf_path" and base.endswith("_nohf"):
                base = base[: -len("_nohf")]
            return base
    sid = st.get("session_id")
    if sid:
        return f"demo_{sid}"
    return f"demo_{uuid.uuid4().hex[:8]}"


def _edited_dir_for(st: dict):
    temp_dir = st.get("temp_dir")
    if not temp_dir:
        temp_dir, _ = create_temp_session_dir()
        st["temp_dir"] = temp_dir
    d = os.path.join(temp_dir, "edited")
    os.makedirs(d, exist_ok=True)
    return d


def _edited_filepath(st: dict, which: str):
    """
    which in {'md','nohf','json'}
    """
    base = _get_base_name_from_result(st)
    if which == "md":
        name = f"{base}.md"
    elif which == "nohf":
        name = f"{base}_nohf.md"
    elif which == "json":
        name = f"{base}.json"
    else:
        raise ValueError(f"unknown edited type: {which}")
    return os.path.join(_edited_dir_for(st), name)


def _save_edited_to_disk(st: dict, which: str, content: str):
    path = _edited_filepath(st, which)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if content is not None else "")
    return path


def _delete_edited_from_disk(st: dict, which: str):
    try:
        path = _edited_filepath(st, which)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _invalidate_export_zip(rid: str):
    st = RESULTS_CACHE.get(rid) or {}
    old = st.get("export_path")
    if old and isinstance(old, str) and os.path.exists(old):
        try:
            os.remove(old)
        except Exception:
            pass
    if "export_path" in st:
        st["export_path"] = None
    RESULTS_CACHE[rid] = st


# ---------------- UI state helpers (per-card) ----------------
def _default_ui_state():
    # 增加 source: '源码' 或 '编辑源码'
    return {"preview": True, "nohf": False, "tab": "md", "source": "源码"}


def _ensure_ui_state(rid):
    st = RESULTS_CACHE.get(rid) or {}
    ui = st.get("ui")
    if not isinstance(ui, dict):
        ui = _default_ui_state()
        st["ui"] = ui
        RESULTS_CACHE[rid] = st
    else:
        # 兼容旧状态缺少新字段
        if "source" not in ui:
            ui["source"] = "源码"
        if "tab" not in ui:
            ui["tab"] = "md"
        if "preview" not in ui:
            ui["preview"] = True
        if "nohf" not in ui:
            ui["nohf"] = False
        RESULTS_CACHE[rid] = st
    return ui


# ---------------- Background worker ----------------
def background_processor():
    while True:
        try:
            task = TASK_QUEUE.get(timeout=1)
        except queue.Empty:
            continue
        if task is None:
            # Important: mark done for sentinel to keep queue counters balanced
            try:
                TASK_QUEUE.task_done()
            finally:
                pass
            break
        rid, filepath, prompt_mode, server_ip, server_port, min_p, max_p, fitz_flag = (
            task
        )
        image = None
        try:
            # Build parser instance for this task
            local_parser = DotsOCRParser(
                ip=server_ip,
                port=int(server_port),
                dpi=200,
                min_pixels=min_p,
                max_pixels=max_p,
            )

            # Read image
            try:
                fp_lower = str(filepath).lower() if isinstance(filepath, str) else ""
                if fitz_flag or fp_lower.endswith(".pdf"):
                    try:
                        import fitz as _fitz

                        doc = _fitz.open(filepath)
                        page = doc.load_page(0)
                        pix = page.get_pixmap()
                        mode = "RGBA" if pix.alpha else "RGB"
                        image = Image.frombytes(
                            mode, (pix.width, pix.height), pix.samples
                        )
                        doc.close()
                    except Exception:
                        image = read_image_v2(filepath)
                else:
                    image = read_image_v2(filepath)
            except Exception as e:
                raise RuntimeError(f"Failed to read input {filepath}: {e}")

            # Parse
            result = parse_image_with_high_level_api(
                local_parser, image, prompt_mode, fitz_preprocess=fitz_flag
            )
            result["status"] = "done"

            # Preserve source/input path but prefer prev.source_path if available
            prev = RESULTS_CACHE.get(rid) or {}

            # Preserve UI state across re-parses/results
            prev_ui = prev.get("ui") if isinstance(prev, dict) else None
            result["ui"] = prev_ui if isinstance(prev_ui, dict) else _default_ui_state()

            if isinstance(prev, dict) and isinstance(prev.get("edits"), dict):
                result["edits"] = dict(prev.get("edits"))

            if isinstance(prev, dict) and prev.get("source_path"):
                result["source_path"] = prev.get("source_path")
            else:
                if isinstance(filepath, str) and os.path.exists(filepath):
                    result["source_path"] = filepath
                else:
                    result["source_path"] = result.get("input_temp_path")

            if isinstance(prev, dict) and prev.get("input_path"):
                result["input_path"] = prev.get("input_path")

            # Commit result
            RESULTS_CACHE[rid] = result

            # Pre-build export zip for first-click download
            try:
                zip_path = ensure_export_ready(rid)
                if zip_path:
                    result = RESULTS_CACHE.get(rid, result)
                    result["export_path"] = zip_path
                    RESULTS_CACHE[rid] = result
            except Exception:
                pass

        except Exception as e:
            # Auto-retry for transient backend errors (e.g., server down temporarily)
            if _is_transient_backend_error(e):
                attempts = RETRY_COUNTS.get(rid, 0)
                if attempts < MAX_AUTO_RETRIES:
                    RETRY_COUNTS[rid] = attempts + 1
                    delay = min(10.0, (RETRY_BACKOFF_BASE**attempts))
                    # keep state pending, annotate attempts
                    prev = RESULTS_CACHE.get(rid, {}) or {}
                    pend_state = dict(prev)
                    pend_state.update(
                        {
                            "status": "pending",
                            "retry_attempts": attempts + 1,
                        }
                    )
                    RESULTS_CACHE[rid] = pend_state

                    # Re-enqueue after delay on a timer to avoid blocking worker
                    def _requeue_later():
                        TASK_QUEUE.put(
                            (
                                rid,
                                filepath,
                                prompt_mode,
                                server_ip,
                                int(server_port),
                                min_p,
                                max_p,
                                fitz_flag,
                            )
                        )

                    threading.Timer(delay, _requeue_later).start()
                    # Do not mark error; move on
                    continue

            # Build a rich error state that preserves re-parse materials
            prev = RESULTS_CACHE.get(rid, {}) or {}
            err_state = dict(prev)  # preserve input_path etc.
            err_state["status"] = "error"
            err_state["md_content"] = classify_parse_failure(e, min_p, max_p)

            # Save a temporary PNG for re-parse if we have an image in memory
            if isinstance(image, Image.Image):
                try:
                    tmp_dir, _sid = create_temp_session_dir()
                    tmp_path = os.path.join(tmp_dir, f"error_input_{rid}.png")
                    image.save(tmp_path, "PNG")
                    err_state["original_image"] = image
                    err_state["input_temp_path"] = tmp_path
                    err_state["temp_dir"] = tmp_dir
                except Exception:
                    err_state["original_image"] = image
            if isinstance(filepath, str) and filepath:
                err_state.setdefault("source_path", filepath)

            # Preserve UI state if missing
            if not isinstance(err_state.get("ui"), dict):
                err_state["ui"] = _default_ui_state()

            RESULTS_CACHE[rid] = err_state
        finally:
            # Mark the non-sentinel task as done
            try:
                # If previous branch already marked sentinel done, skip double mark
                if task is not None:
                    TASK_QUEUE.task_done()
            except Exception:
                pass


def _stop_all_workers():
    """Stop all worker threads gracefully by sending sentinels and joining."""
    global WORKER_THREADS
    with THREAD_LOCK:
        n = len(WORKER_THREADS)
        if n == 0:
            return
        # Send one sentinel per worker
        for _ in range(n):
            TASK_QUEUE.put(None)
        # Join all workers
        for t in WORKER_THREADS:
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
        WORKER_THREADS = []


def _start_workers(count: int):
    """Start exactly `count` worker threads if not already running."""
    global WORKER_THREADS
    with THREAD_LOCK:
        running = len(WORKER_THREADS)
        need = max(0, int(count) - running)
        for _ in range(need):
            t = threading.Thread(target=background_processor, daemon=True)
            t.start()
            WORKER_THREADS.append(t)


def start_background_processor():
    """Ensure at least one worker is running (used by legacy calls)."""
    _start_workers(max(1, MAX_CONCURRENCY))


def set_max_concurrency(n: int):
    """Restart worker pool to match desired concurrency."""
    global MAX_CONCURRENCY
    n = int(n) if isinstance(n, (int, float)) else 1
    if n <= 0:
        n = 1
    MAX_CONCURRENCY = n
    # Restart workers to apply new concurrency
    _stop_all_workers()
    _start_workers(MAX_CONCURRENCY)


# ---------------- Queueing / task helpers ----------------
def _pixel_reasons(min_p, max_p):
    reasons = []
    if min_p < ABS_MIN_PIXELS:
        reasons.append(f"Min Pixels 过小：{min_p}，必须 >= {ABS_MIN_PIXELS}。")
    if max_p > ABS_MAX_PIXELS:
        reasons.append(f"Max Pixels 过大：{max_p}，必须 <= {ABS_MAX_PIXELS}。")
    if min_p >= max_p:
        reasons.append(
            f"像素参数不合法：Min Pixels({min_p}) >= Max Pixels({max_p})，必须满足 Min Pixels < Max Pixels。"
        )
    return reasons


def add_tasks_to_queue(
    file_list, prompt_mode, server_ip, server_port, min_p, max_p, fitz, cur_ids
):
    """Queue uploaded file paths (expects file_list of local file paths or tuples (parse_path, source_path))."""
    if not file_list:
        return cur_ids, "No images uploaded."

    min_p, max_p = _validate_pixels(min_p, max_p)
    start_background_processor()

    ids = list(cur_ids or [])
    skipped = 0
    queued = 0

    for fp in file_list:
        # Normalize: support tuple (parse_path, source_path)
        parse_fp = None
        source_fp = None
        if isinstance(fp, (list, tuple)) and len(fp) >= 1:
            parse_fp = fp[0]
            # If tuple contains original source as second element, use it
            source_fp = fp[1] if len(fp) >= 2 else fp[0]
        else:
            parse_fp = fp
            source_fp = fp

        if isinstance(parse_fp, (list, tuple)):
            parse_fp = parse_fp[0] if len(parse_fp) > 0 else None

        rid = uuid.uuid4().hex[:8]
        ids.append(rid)

        # placeholder with input_path so re-parse works even before parse
        RESULTS_CACHE[rid] = {
            "status": "pending",
            "input_path": parse_fp,
            "source_path": source_fp,
            "ui": _default_ui_state(),  # 初始化每项的独立 UI 状态
        }

        reason = _pixel_reasons(min_p, max_p)
        if reason:
            RESULTS_CACHE[rid] = {
                "status": "error",
                "md_content": "参数越界，未开始解析：\n"
                + "\n".join(f"- {r}" for r in reason)
                + f"\n(当前参数：min_pixels={min_p}, max_pixels={max_p})",
                "input_path": parse_fp,
                "source_path": source_fp,
                "ui": _default_ui_state(),
            }
            skipped += 1
            continue

        TASK_QUEUE.put(
            (
                rid,
                parse_fp,
                prompt_mode,
                server_ip,
                int(server_port),
                min_p,
                max_p,
                fitz,
            )
        )
        queued += 1

    info = f"Queued {queued} item(s)."
    if skipped:
        info += f" Skipped {skipped} due to invalid pixel limits."
    return ids, info


def enqueue_single_reparse(
    rid, reupload_path, prompt_mode, server_ip, server_port, min_p, max_p, fitz
):
    """
    Enqueue a reparse for single result id.
    Path selection priority:
      reupload_path -> result.source_path -> result.input_temp_path -> result.input_path -> result.original_image (dump to temp PNG)
    """
    min_p, max_p = _validate_pixels(min_p, max_p)
    start_background_processor()
    st = RESULTS_CACHE.get(rid, {}) or {}

    # Pixel constraints: if invalid, set error state and return (do not enqueue)
    reason = _pixel_reasons(min_p, max_p)
    if reason:
        new_state = st.copy()
        new_state.update(
            {
                "status": "error",
                "md_content": "参数越界，未开始解析：\n"
                + "\n".join(f"- {r}" for r in reason)
                + f"\n(当前参数：min_pixels={min_p}, max_pixels={max_p})",
            }
        )
        # 保留 UI 状态
        if "ui" not in new_state:
            new_state["ui"] = _default_ui_state()
        RESULTS_CACHE[rid] = new_state
        return

    if isinstance(reupload_path, (tuple, list)):
        reupload_path = reupload_path[0] if len(reupload_path) > 0 else None

    filepath = None
    if reupload_path:
        filepath = reupload_path
    elif st.get("source_path"):
        filepath = st.get("source_path")
    elif st.get("input_temp_path"):
        filepath = st.get("input_temp_path")
    elif st.get("input_path"):
        filepath = st.get("input_path")
    else:
        img = st.get("original_image")
        if isinstance(img, Image.Image):
            tmp_dir, _ = create_temp_session_dir()
            tmp_path = os.path.join(tmp_dir, f"reparse_{rid}.png")
            try:
                img.save(tmp_path, "PNG")
                filepath = tmp_path
            except Exception:
                filepath = None

    if not filepath:
        new_state = st.copy()
        new_state.update(
            {
                "status": "error",
                "md_content": "重解析失败：未找到可用的图片来源。请重新上传图片或检查缓存目录。",
            }
        )
        if "ui" not in new_state:
            new_state["ui"] = _default_ui_state()
        RESULTS_CACHE[rid] = new_state
        return

    new_state = st.copy()
    new_state.update(
        {
            "status": "pending",
            "input_path": filepath,
            "last_used_config": {
                "ip": server_ip,
                "port": int(server_port),
                "min_pixels": min_p,
                "max_pixels": max_p,
                "prompt_mode": prompt_mode,
            },
        }
    )
    # 保留 UI 状态
    if "ui" not in new_state:
        new_state["ui"] = _default_ui_state()
    RESULTS_CACHE[rid] = new_state
    TASK_QUEUE.put(
        (rid, filepath, prompt_mode, server_ip, int(server_port), min_p, max_p, fitz)
    )


def delete_one(ids, rid, tick):
    new_ids = [x for x in (ids or []) if x != rid]
    st = RESULTS_CACHE.get(rid)
    temp_dir = st.get("temp_dir") if st else None
    if rid in RESULTS_CACHE:
        del RESULTS_CACHE[rid]
    if rid in RETRY_COUNTS:
        del RETRY_COUNTS[rid]
    purge_queue(rid)
    if temp_dir and os.path.exists(temp_dir):
        threading.Thread(
            target=lambda: shutil.rmtree(temp_dir, ignore_errors=True), daemon=True
        ).start()
    return new_ids, int(tick or 0) + 1


# ---------------- Gradio UI ----------------
def create_gradio_interface():
    css = """
    /* basic theme */
    :root { --bg:#0b1220; --card:#111827; --muted:#9ca3af; --accent:#FF576D; --text:#e5e7eb; }
    body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
    .result-card { background: var(--card); border:1px solid #1f2937; border-radius:8px; padding:10px; margin-bottom:12px; }
    .muted { color: var(--muted); font-size:0.9em; }

    /* skeleton shimmer */
    .skeleton { position:relative; overflow:hidden; background:#0f172a; border-radius:6px; }
    .skeleton::after {
      content:""; position:absolute; inset:0; transform:translateX(-100%);
      background:linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.06), rgba(255,255,255,0));
      animation:shimmer 1.2s infinite;
    }
    @keyframes shimmer { 100% { transform:translateX(100%);} }

    /* Hide unwanted footer/buttons (robust selectors) */
    footer, .footer, #footer, footer[role="contentinfo"] { display:none !important; }
    [aria-label="Use via API"], [aria-label*="API"], [title*="API"], a[href*="/api"], a[href*="api_docs"], a[href*="gradio.app"] { display:none !important; }
    button[aria-label="Settings"], button[aria-label*="设置"], [aria-label="Built with Gradio"] { display:none !important; }

        /* Script log area: single inner scrollbar on <pre>, outer container hidden overflow */
        .script-log { max-height: 260px; overflow: hidden; border:1px solid #1f2937; border-radius:6px; padding:0; }
        .script-log pre {
            max-height: 260px;
            overflow: auto;
            margin: 0;
            padding: 6px;
            background: transparent;
            scrollbar-width: thin; /* Firefox */
            scrollbar-color: rgba(255,255,255,0.2) transparent;
        }
        .script-log pre::-webkit-scrollbar { width: 6px; height: 6px; }
        .script-log pre::-webkit-scrollbar-track { background: transparent; }
        .script-log pre::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
        .script-log pre:hover::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.25); }
    """

    with gr.Blocks(css=css, title="dots.ocr") as demo:
        # Left column controls
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Multiple Images",
                    type="filepath",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                )
                # Filter out the unwanted 'prompt_grounding_ocr' mode
                allowed_modes = [
                    m
                    for m in dict_promptmode_to_prompt.keys()
                    if m != "prompt_grounding_ocr"
                ]
                if not allowed_modes:
                    allowed_modes = list(dict_promptmode_to_prompt.keys())
                prompt_mode = gr.Dropdown(
                    label="Prompt Mode",
                    choices=allowed_modes,
                    value=allowed_modes[0],
                )
                prompt_display = gr.Textbox(
                    label="Prompt Preview",
                    value=dict_promptmode_to_prompt[allowed_modes[0]],
                    interactive=False,
                    lines=4,
                )

                with gr.Row():
                    parse_btn = gr.Button("🔍 Parse", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

                with gr.Accordion("Advanced Config", open=False):
                    fitz_preprocess = gr.Checkbox(label="fitz_preprocess", value=True)
                    server_ip = gr.Textbox(
                        label="Server IP", value=DEFAULT_CONFIG["ip"]
                    )
                    server_port = gr.Number(
                        label="Port", value=DEFAULT_CONFIG["port_vllm"], precision=0
                    )
                    min_pixels = gr.Number(
                        label="Min Pixels", value=DEFAULT_CONFIG["min_pixels"]
                    )
                    max_pixels = gr.Number(
                        label="Max Pixels", value=DEFAULT_CONFIG["max_pixels"]
                    )
                    concurrency = gr.Number(
                        label="Max Concurrency",
                        value=MAX_CONCURRENCY,  # 与实际生效的后台并发保持一致（支持刷新后保持）
                        precision=0,
                        interactive=True,
                    )
                    confirm_delete = gr.Checkbox(
                        label="删除前确认（推荐）", value=True, interactive=True
                    )

            # Right column: results & actions
            with gr.Column(scale=5):
                info_display = gr.Markdown("Waiting...", elem_id="info_box")
                ids_state = gr.State(value=[])
                store_tick = gr.State(value=0)
                render_bump = gr.State(value=0)  # 仅用于在状态变化时触发结果重渲染
                confirm_delete_state = gr.State(value=True)
                confirm_delete.change(
                    lambda v: v, inputs=[confirm_delete], outputs=[confirm_delete_state]
                )

                progress_timer = gr.Timer(1.0)

                # Actions 面板（多选）
                with gr.Accordion("Actions", open=False):
                    selected_group = gr.CheckboxGroup(
                        label="Select Items", choices=[], value=[], interactive=True
                    )
                    with gr.Row():
                        select_all_btn = gr.Button("全选")
                        clear_sel_btn = gr.Button("清空选择")
                    with gr.Row():
                        bulk_reparse_btn = gr.Button("🔁 重解析所选")
                        delete_selected_btn = gr.Button("🗑️ 删除所选", variant="stop")
                        export_selected_btn = gr.DownloadButton("📦 导出所选")
                    # 高级脚本导出
                    with gr.Accordion("高级脚本", open=False):
                        gr.Markdown(
                            "在下方编辑并运行自定义 Python 脚本以自由处理当前解析结果并导出为任意目录/文件结构。"
                            "<br/>脚本将在受限环境中执行，可通过 api 对象访问只读数据与构建导出压缩包。",
                            elem_classes=["muted"],
                        )
                        script_code = gr.Code(
                            label="Python 脚本",
                            language="python",
                            value=DEFAULT_SCRIPT_TEMPLATE,
                            lines=24,
                            interactive=True,
                        )
                        with gr.Row():
                            run_script_btn = gr.Button("▶ 运行脚本", variant="primary")
                            script_download_btn = gr.DownloadButton("📦 下载脚本输出")
                        script_status = gr.HTML("")
                        script_log = gr.Markdown(
                            "", elem_id="script_log", elem_classes=["script-log"]
                        )

                        # 流式执行脚本：实时打印日志与运行状态，并在完成后绑定下载按钮
                        run_script_btn.click(
                            run_user_script_stream,
                            inputs=[script_code, ids_state],
                            outputs=[script_download_btn, script_status, script_log],
                            show_progress="hidden",
                        )
                    # 批量删除确认面板
                    with gr.Row(visible=False) as bulk_delete_confirm_panel:
                        gr.Markdown(
                            "确认删除所选结果？该操作不可恢复。",
                            elem_classes=["muted"],
                        )
                        bulk_confirm_delete_btn = gr.Button("确认删除", variant="stop")
                        bulk_cancel_delete_btn = gr.Button("取消")

                # Render results dynamically
                @gr.render(inputs=[ids_state, render_bump])
                def render_results(ids, _bump):
                    if not ids:
                        return gr.Markdown("No results yet.")
                    with gr.Column():
                        for idx, rid in enumerate(ids):
                            data = RESULTS_CACHE.get(rid, {}) or {}
                            status = data.get("status", "pending")

                            # 确保每张卡都有独立 UI 状态（并写回缓存，保证后续使用）
                            ui = _ensure_ui_state(rid)
                            preview_on = bool(ui.get("preview", True))
                            nohf_on = bool(ui.get("nohf", False))
                            active_tab = ui.get("tab", "md")
                            if active_tab not in ("md", "json"):
                                active_tab = "md"
                            source_sel = ui.get("source", "源码")
                            if source_sel not in ("源码", "编辑源码"):
                                source_sel = "源码"

                            with gr.Column(
                                elem_classes=["result-card"], elem_id=f"card-{rid}"
                            ):
                                with gr.Row():
                                    gr.Markdown(
                                        f"### Result {idx+1} <span class='muted'>RID: {rid}</span>"
                                    )

                                if status == "error":
                                    gr.Markdown(
                                        f"⚠️ 解析失败：\n\n{data.get('md_content','Unknown error')}",
                                        elem_classes=["muted"],
                                    )

                                if status == "done":
                                    orig_img = data.get("original_image")
                                    layout_img = data.get("layout_image")
                                    with gr.Row():
                                        gr.Image(
                                            value=orig_img, label="Original", height=300
                                        )
                                        gr.Image(
                                            value=layout_img, label="Layout", height=300
                                        )
                                elif status == "pending":
                                    with gr.Row():
                                        gr.HTML(
                                            "<div class='skeleton' style='width:100%;height:300px;'></div>"
                                        )
                                        gr.HTML(
                                            "<div class='skeleton' style='width:100%;height:300px;'></div>"
                                        )

                                # badges
                                with gr.Row():
                                    badge_md = gr.HTML(
                                        f"<span class='muted'>MD: {'Preview' if preview_on else 'Source'}</span>"
                                    )
                                    badge_nohf = gr.HTML(
                                        f"<span class='muted'>NOHF: {'On' if nohf_on else 'Off'}</span>"
                                    )

                                # controls
                                with gr.Row():
                                    rid_box = gr.Textbox(value=rid, visible=False)
                                    preview_cb = gr.Checkbox(
                                        label="Preview Markdown",
                                        value=preview_on,
                                    )
                                    nohf_cb = gr.Checkbox(label="NOHF", value=nohf_on)

                                # 视图切换
                                selected_label = (
                                    "Markdown" if active_tab == "md" else "JSON"
                                )
                                with gr.Row():
                                    view_radio = gr.Radio(
                                        label="视图",
                                        choices=["Markdown", "JSON"],
                                        value=selected_label,
                                    )

                                # 内容来源（仅完成状态可用）
                                with gr.Row():
                                    source_radio = gr.Radio(
                                        label="内容来源",
                                        choices=["源码", "编辑源码"],
                                        value=source_sel,
                                        interactive=True,
                                        visible=(status == "done"),
                                    )

                                # 内容获取助手
                                def _get_texts(rid_value, nohf_flag):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    md_orig = st.get("md_content") or ""
                                    md_nohf_orig = st.get("md_content_nohf") or ""
                                    md_current_orig = (
                                        md_nohf_orig if nohf_flag else md_orig
                                    )
                                    edits = st.get("edits") or {}
                                    md_edit = (
                                        edits.get("nohf")
                                        if nohf_flag
                                        else edits.get("md")
                                    )
                                    if md_edit is None:
                                        md_edit = md_current_orig
                                    json_orig = st.get("json_code") or ""
                                    json_edit = edits.get("json")
                                    if json_edit is None:
                                        json_edit = json_orig
                                    return (
                                        md_current_orig,
                                        md_edit,
                                        json_orig,
                                        json_edit,
                                    )

                                (
                                    md_orig_val,
                                    md_edit_val,
                                    json_orig_val,
                                    json_edit_val,
                                ) = _get_texts(rid, nohf_on)
                                is_md_init = selected_label == "Markdown"
                                use_edit_init = source_sel == "编辑源码"

                                # 单一预览组件（Markdown 用）
                                md_preview = gr.Markdown(
                                    value=(
                                        md_edit_val if use_edit_init else md_orig_val
                                    ),
                                    visible=(
                                        status == "done" and is_md_init and preview_on
                                    ),
                                )
                                # 原始源码（只读）
                                md_code_orig = gr.Code(
                                    language="markdown",
                                    value=md_orig_val,
                                    interactive=False,
                                    visible=(
                                        status == "done"
                                        and is_md_init
                                        and (not preview_on)
                                        and (not use_edit_init)
                                    ),
                                )
                                # 编辑源码（可编辑、自动保存）
                                md_code_edit = gr.Code(
                                    language="markdown",
                                    value=md_edit_val,
                                    interactive=True,
                                    visible=(
                                        status == "done"
                                        and is_md_init
                                        and (not preview_on)
                                        and use_edit_init
                                    ),
                                )

                                # JSON（原始与编辑）
                                json_code_orig = gr.Code(
                                    language="json",
                                    value=json_orig_val,
                                    interactive=False,
                                    visible=(
                                        status == "done"
                                        and (not is_md_init)
                                        and (not use_edit_init)
                                    ),
                                )
                                json_code_edit = gr.Code(
                                    language="json",
                                    value=json_edit_val,
                                    interactive=True,
                                    visible=(
                                        status == "done"
                                        and (not is_md_init)
                                        and use_edit_init
                                    ),
                                )

                                # 仅编辑模式显示
                                restore_btn = gr.Button(
                                    "还原当前内容",
                                    visible=(status == "done" and use_edit_init),
                                )

                                # 统一可见性/内容更新
                                def _apply_all(
                                    preview, use_nohf, view_label, src_label, rid_value
                                ):
                                    preview = bool(preview)
                                    use_nohf = bool(use_nohf)
                                    is_md = str(view_label) == "Markdown"
                                    use_edit = str(src_label) == "编辑源码"

                                    # 写回 UI 状态
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    ui0 = dict(st.get("ui") or _default_ui_state())
                                    ui0["preview"] = preview
                                    ui0["nohf"] = use_nohf
                                    ui0["tab"] = "md" if is_md else "json"
                                    ui0["source"] = "编辑源码" if use_edit else "源码"
                                    st["ui"] = ui0
                                    RESULTS_CACHE[rid_value] = st

                                    md_o, md_e, j_o, j_e = _get_texts(
                                        rid_value, use_nohf
                                    )
                                    return (
                                        gr.update(
                                            value=f"<span class='muted'>MD: {'Preview' if preview else 'Source'}</span>"
                                        ),
                                        gr.update(
                                            value=f"<span class='muted'>NOHF: {'On' if use_nohf else 'Off'}</span>"
                                        ),
                                        gr.update(
                                            value=(md_e if use_edit else md_o),
                                            visible=(is_md and preview),
                                        ),
                                        gr.update(
                                            value=md_o,
                                            visible=(
                                                is_md
                                                and (not preview)
                                                and (not use_edit)
                                            ),
                                        ),
                                        gr.update(
                                            value=md_e,
                                            visible=(
                                                is_md and (not preview) and use_edit
                                            ),
                                        ),
                                        gr.update(
                                            value=j_o,
                                            visible=(not is_md and (not use_edit)),
                                        ),
                                        gr.update(
                                            value=j_e, visible=(not is_md and use_edit)
                                        ),
                                        gr.update(visible=use_edit),
                                    )

                                # 绑定控制项变化：预览、NOHF、视图、来源
                                preview_cb.change(
                                    _apply_all,
                                    inputs=[
                                        preview_cb,
                                        nohf_cb,
                                        view_radio,
                                        source_radio,
                                        rid_box,
                                    ],
                                    outputs=[
                                        badge_md,
                                        badge_nohf,
                                        md_preview,
                                        md_code_orig,
                                        md_code_edit,
                                        json_code_orig,
                                        json_code_edit,
                                        restore_btn,
                                    ],
                                    show_progress="hidden",
                                )
                                nohf_cb.change(
                                    _apply_all,
                                    inputs=[
                                        preview_cb,
                                        nohf_cb,
                                        view_radio,
                                        source_radio,
                                        rid_box,
                                    ],
                                    outputs=[
                                        badge_md,
                                        badge_nohf,
                                        md_preview,
                                        md_code_orig,
                                        md_code_edit,
                                        json_code_orig,
                                        json_code_edit,
                                        restore_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                def _on_view_change(
                                    view_label,
                                    rid_value,
                                    preview_flag,
                                    nohf_flag,
                                    src_label,
                                ):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    ui0 = dict(st.get("ui") or _default_ui_state())
                                    ui0["tab"] = (
                                        "md"
                                        if str(view_label) == "Markdown"
                                        else "json"
                                    )
                                    st["ui"] = ui0
                                    RESULTS_CACHE[rid_value] = st
                                    return _apply_all(
                                        preview_flag,
                                        nohf_flag,
                                        view_label,
                                        src_label,
                                        rid_value,
                                    )

                                view_radio.change(
                                    _on_view_change,
                                    inputs=[
                                        view_radio,
                                        rid_box,
                                        preview_cb,
                                        nohf_cb,
                                        source_radio,
                                    ],
                                    outputs=[
                                        badge_md,
                                        badge_nohf,
                                        md_preview,
                                        md_code_orig,
                                        md_code_edit,
                                        json_code_orig,
                                        json_code_edit,
                                        restore_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                def _on_source_change(
                                    src_label,
                                    rid_value,
                                    preview_flag,
                                    nohf_flag,
                                    view_label,
                                ):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    ui0 = dict(st.get("ui") or _default_ui_state())
                                    ui0["source"] = (
                                        "编辑源码"
                                        if str(src_label) == "编辑源码"
                                        else "源码"
                                    )
                                    st["ui"] = ui0
                                    RESULTS_CACHE[rid_value] = st
                                    return _apply_all(
                                        preview_flag,
                                        nohf_flag,
                                        view_label,
                                        src_label,
                                        rid_value,
                                    )

                                source_radio.change(
                                    _on_source_change,
                                    inputs=[
                                        source_radio,
                                        rid_box,
                                        preview_cb,
                                        nohf_cb,
                                        view_radio,
                                    ],
                                    outputs=[
                                        badge_md,
                                        badge_nohf,
                                        md_preview,
                                        md_code_orig,
                                        md_code_edit,
                                        json_code_orig,
                                        json_code_edit,
                                        restore_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                # Action buttons per-card
                                with gr.Row():
                                    reparse_btn = gr.Button(
                                        "🔁 重新解析",
                                        interactive=(status in ("done", "error")),
                                    )
                                    export_btn = gr.DownloadButton(
                                        "📦 导出",
                                        interactive=(status == "done"),
                                        value=(
                                            data.get("export_path")
                                            if status == "done"
                                            else None
                                        ),
                                    )
                                    delete_btn = gr.Button("🗑️ 删除", variant="stop")

                                # 自动保存（编辑器变更即写盘 + 刷新导出 + 可能的 Markdown 预览）
                                def _save_md_edit(
                                    val,
                                    rid_value,
                                    nohf_flag,
                                    preview_flag,
                                    view_label,
                                    src_label,
                                    ids,
                                    selected_labels,
                                ):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    if st.get("status") != "done":
                                        # 同步“导出所选”以防其它项在编辑（极少见）
                                        path_sel = export_selected_rids(
                                            ids, selected_labels
                                        )
                                        return (
                                            gr.update(),
                                            gr.update(),
                                            gr.update(value=path_sel),
                                        )
                                    which = "nohf" if bool(nohf_flag) else "md"
                                    edits = dict(st.get("edits") or {})
                                    edits[which] = val or ""
                                    st["edits"] = edits
                                    RESULTS_CACHE[rid_value] = st
                                    try:
                                        _save_edited_to_disk(st, which, val or "")
                                    except Exception:
                                        pass
                                    _invalidate_export_zip(rid_value)
                                    new_zip = ensure_export_ready(rid_value)

                                    # 刷新“导出所选”
                                    path_sel = export_selected_rids(
                                        ids, selected_labels
                                    )

                                    # 若当前正处于 Markdown/预览/编辑模式，则更新预览内容
                                    is_md = str(view_label) == "Markdown"
                                    use_edit = str(src_label) == "编辑源码"
                                    if is_md and use_edit and bool(preview_flag):
                                        return (
                                            gr.update(value=val or ""),
                                            gr.update(value=new_zip),
                                            gr.update(value=path_sel),
                                        )
                                    return (
                                        gr.update(),
                                        gr.update(value=new_zip),
                                        gr.update(value=path_sel),
                                    )

                                md_code_edit.change(
                                    _save_md_edit,
                                    inputs=[
                                        md_code_edit,
                                        rid_box,
                                        nohf_cb,
                                        preview_cb,
                                        view_radio,
                                        source_radio,
                                        ids_state,
                                        selected_group,
                                    ],
                                    outputs=[
                                        md_preview,
                                        export_btn,
                                        export_selected_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                def _save_json_edit(
                                    val, rid_value, ids, selected_labels
                                ):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    if st.get("status") != "done":
                                        path_sel = export_selected_rids(
                                            ids, selected_labels
                                        )
                                        return gr.update(), gr.update(value=path_sel)
                                    edits = dict(st.get("edits") or {})
                                    edits["json"] = val or ""
                                    st["edits"] = edits
                                    RESULTS_CACHE[rid_value] = st
                                    try:
                                        _save_edited_to_disk(st, "json", val or "")
                                    except Exception:
                                        pass
                                    _invalidate_export_zip(rid_value)
                                    new_zip = ensure_export_ready(rid_value)
                                    path_sel = export_selected_rids(
                                        ids, selected_labels
                                    )
                                    return gr.update(value=new_zip), gr.update(
                                        value=path_sel
                                    )

                                json_code_edit.change(
                                    _save_json_edit,
                                    inputs=[
                                        json_code_edit,
                                        rid_box,
                                        ids_state,
                                        selected_group,
                                    ],
                                    outputs=[export_btn, export_selected_btn],
                                    show_progress="hidden",
                                )

                                # 还原当前内容
                                def _restore_current(
                                    src_label,
                                    rid_value,
                                    nohf_flag,
                                    preview_flag,
                                    view_label,
                                    ids,
                                    selected_labels,
                                ):
                                    st = RESULTS_CACHE.get(rid_value, {}) or {}
                                    which = (
                                        "json"
                                        if str(view_label) == "JSON"
                                        else ("nohf" if bool(nohf_flag) else "md")
                                    )
                                    # 删除编辑版
                                    edits = dict(st.get("edits") or {})
                                    if which in edits:
                                        edits.pop(which, None)
                                        st["edits"] = edits
                                    RESULTS_CACHE[rid_value] = st
                                    try:
                                        _delete_edited_from_disk(st, which)
                                    except Exception:
                                        pass
                                    # 重新取原始内容
                                    md_o, md_e, j_o, j_e = _get_texts(
                                        rid_value, bool(nohf_flag)
                                    )
                                    # 刷新导出
                                    _invalidate_export_zip(rid_value)
                                    new_zip = ensure_export_ready(rid_value)
                                    path_sel = export_selected_rids(
                                        ids, selected_labels
                                    )
                                    # 更新编辑器与预览
                                    up_md_editor = (
                                        gr.update(value=md_o)
                                        if which in ("md", "nohf")
                                        else gr.update()
                                    )
                                    up_json_editor = (
                                        gr.update(value=j_o)
                                        if which == "json"
                                        else gr.update()
                                    )
                                    is_md = str(view_label) == "Markdown"
                                    use_edit = str(src_label) == "编辑源码"
                                    up_preview = (
                                        gr.update(value=(md_e if use_edit else md_o))
                                        if is_md and bool(preview_flag)
                                        else gr.update()
                                    )
                                    return (
                                        up_md_editor,
                                        up_json_editor,
                                        up_preview,
                                        gr.update(value=new_zip),
                                        gr.update(value=path_sel),
                                    )

                                restore_btn.click(
                                    _restore_current,
                                    inputs=[
                                        source_radio,
                                        rid_box,
                                        nohf_cb,
                                        preview_cb,
                                        view_radio,
                                        ids_state,
                                        selected_group,
                                    ],
                                    outputs=[
                                        md_code_edit,
                                        json_code_edit,
                                        md_preview,
                                        export_btn,
                                        export_selected_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                # Reparse panel (collapsed)
                                with gr.Column(visible=False) as reparse_panel:
                                    gr.Markdown("**重解析**")
                                    with gr.Row():
                                        reparse_current_btn = gr.Button(
                                            "基于当前图片直接重解析", variant="primary"
                                        )

                                # Delete confirm panel (collapsed)
                                with gr.Row(visible=False) as delete_confirm_panel:
                                    gr.Markdown(
                                        "确认删除该结果？该操作不可恢复。",
                                        elem_classes=["muted"],
                                    )
                                    confirm_delete_btn = gr.Button(
                                        "确认删除", variant="stop"
                                    )
                                    cancel_delete_btn = gr.Button("取消")

                                # 绑定其他交互
                                reparse_btn.click(
                                    lambda: gr.update(visible=True),
                                    outputs=[reparse_panel],
                                    show_progress="hidden",
                                )

                                def _start_reparse_current(
                                    rid_value,
                                    p_mode,
                                    ip_addr,
                                    port_val,
                                    minp,
                                    maxp,
                                    fitz_flag,
                                    tick,
                                    ids,
                                    selected_labels,
                                ):
                                    try:
                                        enqueue_single_reparse(
                                            rid_value,
                                            None,
                                            p_mode,
                                            ip_addr,
                                            int(port_val),
                                            int(minp),
                                            int(maxp),
                                            fitz_flag,
                                        )
                                        # 重建“导出所选”
                                        path_sel = export_selected_rids(
                                            ids, selected_labels
                                        )
                                        return (
                                            int(tick or 0) + 1,
                                            gr.update(visible=False),
                                            gr.update(value=path_sel),
                                        )
                                    except Exception as e:
                                        RESULTS_CACHE[rid_value] = {
                                            "status": "error",
                                            "md_content": f"Reparse error: {e}",
                                            # 保留 UI 状态
                                            "ui": _ensure_ui_state(rid_value),
                                        }
                                        path_sel = export_selected_rids(
                                            ids, selected_labels
                                        )
                                        return (
                                            int(tick or 0) + 1,
                                            gr.update(visible=False),
                                            gr.update(value=path_sel),
                                        )

                                reparse_current_btn.click(
                                    _start_reparse_current,
                                    inputs=[
                                        rid_box,
                                        prompt_mode,
                                        server_ip,
                                        server_port,
                                        min_pixels,
                                        max_pixels,
                                        fitz_preprocess,
                                        store_tick,
                                        ids_state,
                                        selected_group,
                                    ],
                                    outputs=[
                                        store_tick,
                                        reparse_panel,
                                        export_selected_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                def _on_delete_click(
                                    rid_value, ids, need_confirm, tick
                                ):
                                    # 如果需要确认，仅展开确认面板，不修改选择框/导出按钮
                                    if need_confirm:
                                        return (
                                            gr.update(visible=True),
                                            ids,
                                            tick,
                                            gr.update(),  # selected_group 不变
                                            gr.update(),  # export button 不变
                                        )
                                    # 直接删除：更新 ids/tick，并同步 Actions 的选择项与导出按钮
                                    new_ids, new_tick = delete_one(ids, rid_value, tick)
                                    choices = [
                                        f"Result {i+1}"
                                        for i in range(len(new_ids or []))
                                    ]
                                    return (
                                        gr.update(visible=False),
                                        new_ids,
                                        new_tick,
                                        gr.update(choices=choices, value=[]),
                                        gr.update(value=None),  # 清空导出
                                    )

                                # 单卡删除输出同步 selected_group 与 export_selected_btn
                                delete_btn.click(
                                    _on_delete_click,
                                    inputs=[
                                        rid_box,
                                        ids_state,
                                        confirm_delete_state,
                                        store_tick,
                                    ],
                                    outputs=[
                                        delete_confirm_panel,
                                        ids_state,
                                        store_tick,
                                        selected_group,
                                        export_selected_btn,
                                    ],
                                    show_progress="hidden",
                                )

                                def _confirm_delete(rid_value, ids, tick):
                                    new_ids, new_tick = delete_one(ids, rid_value, tick)
                                    choices = [
                                        f"Result {i+1}"
                                        for i in range(len(new_ids or []))
                                    ]
                                    return (
                                        new_ids,
                                        new_tick,
                                        gr.update(visible=False),
                                        gr.update(choices=choices, value=[]),
                                        gr.update(value=None),
                                    )

                                # 确认删除后同步 selected_group 与 export_selected_btn
                                confirm_delete_btn.click(
                                    _confirm_delete,
                                    inputs=[rid_box, ids_state, store_tick],
                                    outputs=[
                                        ids_state,
                                        store_tick,
                                        delete_confirm_panel,
                                        selected_group,
                                        export_selected_btn,
                                    ],
                                    show_progress="hidden",
                                )
                                cancel_delete_btn.click(
                                    lambda: gr.update(visible=False),
                                    outputs=[delete_confirm_panel],
                                    show_progress="hidden",
                                )

                # Top-level events
                def _on_prompt_mode_change(m):
                    return dict_promptmode_to_prompt.get(m, "")

                prompt_mode.change(
                    fn=_on_prompt_mode_change,
                    inputs=[prompt_mode],
                    outputs=[prompt_display],
                    show_progress="hidden",
                )

                def process_images_simple(
                    file_list,
                    p_mode,
                    server_ip_val,
                    server_port_val,
                    min_p_val,
                    max_p_val,
                    fitz_val,
                    cur_ids,
                    tick,
                ):
                    """
                    Process images with selected prompt mode. Grounding mode is removed; all files go through normal path.
                    """
                    minp, maxp = _validate_pixels(min_p_val, max_p_val)
                    _set_parser_config(server_ip_val, server_port_val, minp, maxp)

                    # normalize file_list (gradio file element may pass nested lists)
                    files = []
                    if not file_list:
                        return (
                            gr.update(value=None),
                            gr.update(value="No files uploaded."),
                            cur_ids,
                            tick,
                            gr.update(choices=[], value=[]),
                            gr.update(value=None),  # 清空导出
                        )

                    # build normalized list
                    for f in file_list:
                        if isinstance(f, (list, tuple)):
                            files.append(f[0] if len(f) > 0 else None)
                        else:
                            files.append(f)

                    # Normal path: queue originals
                    new_ids, info = add_tasks_to_queue(
                        files,
                        p_mode,
                        server_ip_val,
                        server_port_val,
                        minp,
                        maxp,
                        fitz_val,
                        cur_ids,
                    )
                    # Update checkbox group choices
                    choices = [f"Result {i+1}" for i in range(len(new_ids or []))]
                    return (
                        gr.update(value=None),
                        gr.update(value=info),
                        new_ids,
                        int(tick or 0) + 1,
                        gr.update(choices=choices, value=[]),
                        gr.update(value=None),  # 清空导出
                    )

                parse_btn.click(
                    fn=process_images_simple,
                    inputs=[
                        file_input,
                        prompt_mode,
                        server_ip,
                        server_port,
                        min_pixels,
                        max_pixels,
                        fitz_preprocess,
                        ids_state,
                        store_tick,
                    ],
                    outputs=[
                        file_input,
                        info_display,
                        ids_state,
                        store_tick,
                        selected_group,
                        export_selected_btn,
                    ],
                    show_progress="hidden",
                )

                # Concurrency change handler: apply immediately
                def _on_concurrency_change(n):
                    try:
                        set_max_concurrency(int(n))
                        return gr.update(value=f"并发已设置为 {int(n)}。")
                    except Exception as e:
                        return gr.update(value=f"设置并发失败：{e}")

                concurrency.change(
                    _on_concurrency_change,
                    inputs=[concurrency],
                    outputs=[info_display],
                    show_progress="hidden",
                )

                # 会话加载时同步 UI 与当前真实并发（解决刷新后 UI 值与实际不一致）
                def _sync_concurrency_on_session_load():
                    try:
                        # 如有需要，补齐 worker 到目标并发数（不会减少已有线程）
                        _start_workers(max(1, MAX_CONCURRENCY))
                        return (
                            gr.update(value=int(MAX_CONCURRENCY)),
                            gr.update(
                                value=f"已同步当前并发为 {int(MAX_CONCURRENCY)}。"
                            ),
                        )
                    except Exception as e:
                        return (
                            gr.update(value=int(MAX_CONCURRENCY)),
                            gr.update(value=f"同步并发时发生异常：{e}"),
                        )

                demo.load(
                    _sync_concurrency_on_session_load,
                    inputs=None,
                    outputs=[concurrency, info_display],
                )

                # 生成导出 ZIP（基于当前选择），用于首次点击即可下载
                def _update_export_for_selection(ids, selected_labels):
                    path = export_selected_rids(ids, selected_labels)
                    return gr.update(
                        value=path if path and os.path.exists(path) else None
                    )

                # Actions: 全选/清空
                def _select_all(ids):
                    choices = [f"Result {i+1}" for i in range(len(ids or []))]
                    # 预生成 zip
                    path = export_selected_rids(ids, choices)
                    return (
                        gr.update(choices=choices, value=choices),
                        gr.update(
                            value=path if path and os.path.exists(path) else None
                        ),
                    )

                def _clear_selection(ids):
                    choices = [f"Result {i+1}" for i in range(len(ids or []))]
                    return (
                        gr.update(choices=choices, value=[]),
                        gr.update(value=None),
                    )

                select_all_btn.click(
                    _select_all,
                    inputs=[ids_state],
                    outputs=[selected_group, export_selected_btn],
                    show_progress="hidden",
                )
                clear_sel_btn.click(
                    _clear_selection,
                    inputs=[ids_state],
                    outputs=[selected_group, export_selected_btn],
                    show_progress="hidden",
                )

                # 当用户手动变更选择时，预构建导出 zip 并绑定到按钮
                selected_group.change(
                    _update_export_for_selection,
                    inputs=[ids_state, selected_group],
                    outputs=[export_selected_btn],
                    show_progress="hidden",
                )

                # Actions: 批量重解析（基于当前图片）
                def bulk_reparse(
                    selected_labels, ids, p_mode, ip, port, minp, maxp, fitz, tick
                ):
                    if not ids or not selected_labels:
                        path_sel = export_selected_rids(ids, selected_labels)
                        return (
                            gr.update(value="未选择任何结果。"),
                            int(tick or 0),
                            gr.update(value=path_sel),
                        )
                    # Map labels -> rids
                    count = 0
                    for label in selected_labels:
                        try:
                            idx = int(str(label).split()[-1]) - 1
                            rid = ids[idx]
                            enqueue_single_reparse(
                                rid,
                                None,
                                p_mode,
                                ip,
                                int(port),
                                int(minp),
                                int(maxp),
                                fitz,
                            )
                            count += 1
                        except Exception:
                            continue
                    path_sel = export_selected_rids(ids, selected_labels)
                    return (
                        gr.update(value=f"已触发 {count} 个重解析任务。"),
                        int(tick or 0) + 1,
                        gr.update(value=path_sel),
                    )

                bulk_reparse_btn.click(
                    bulk_reparse,
                    inputs=[
                        selected_group,
                        ids_state,
                        prompt_mode,
                        server_ip,
                        server_port,
                        min_pixels,
                        max_pixels,
                        fitz_preprocess,
                        store_tick,
                    ],
                    outputs=[info_display, store_tick, export_selected_btn],
                    show_progress="hidden",
                )

                # Actions: 删除所选（尊重“删除前确认”）
                def delete_selected_action(ids, selected_labels, tick):
                    # 先从“原始 ids 列表”解析出要删除的 rid 列表，避免索引随删除而错位
                    if not ids or not selected_labels:
                        choices = [f"Result {i+1}" for i in range(len(ids or []))]
                        return (
                            ids,
                            int(tick or 0),
                            gr.update(choices=choices, value=[]),
                            gr.update(value=None),
                        )
                    # 解析 label -> index（去重、过滤非法）
                    sel_indices = []
                    for label in selected_labels:
                        try:
                            idx = int(str(label).split()[-1]) - 1
                            if 0 <= idx < len(ids):
                                sel_indices.append(idx)
                        except Exception:
                            continue
                    if not sel_indices:
                        choices = [f"Result {i+1}" for i in range(len(ids or []))]
                        return (
                            ids,
                            int(tick or 0),
                            gr.update(choices=choices, value=[]),
                            gr.update(value=None),
                        )
                    sel_indices = sorted(set(sel_indices))
                    rids_to_delete = [ids[i] for i in sel_indices]

                    new_ids = list(ids)
                    new_tick = int(tick or 0)
                    # 基于 rid 删除，避免受索引变化影响
                    for rid in rids_to_delete:
                        new_ids, new_tick = delete_one(new_ids, rid, new_tick)

                    choices = [f"Result {i+1}" for i in range(len(new_ids or []))]
                    return (
                        new_ids,
                        new_tick,
                        gr.update(choices=choices, value=[]),
                        gr.update(value=None),
                    )

                def _on_bulk_delete_click(ids, selected_labels, need_confirm, tick):
                    if need_confirm:
                        # 展示确认面板，不改动任何选择与导出
                        return (
                            gr.update(visible=True),
                            ids,
                            tick,
                            gr.update(),
                            gr.update(),
                        )
                    # 直接删除并隐藏确认面板
                    new_ids, new_tick, sel_update, export_update = (
                        delete_selected_action(ids, selected_labels, tick)
                    )
                    return (
                        gr.update(visible=False),
                        new_ids,
                        new_tick,
                        sel_update,
                        export_update,
                    )

                delete_selected_btn.click(
                    _on_bulk_delete_click,
                    inputs=[
                        ids_state,
                        selected_group,
                        confirm_delete_state,
                        store_tick,
                    ],
                    outputs=[
                        bulk_delete_confirm_panel,
                        ids_state,
                        store_tick,
                        selected_group,
                        export_selected_btn,
                    ],
                    show_progress="hidden",
                )

                def _bulk_confirm_delete(ids, selected_labels, tick):
                    new_ids, new_tick, sel_update, export_update = (
                        delete_selected_action(ids, selected_labels, tick)
                    )
                    return (
                        new_ids,
                        new_tick,
                        sel_update,
                        export_update,
                        gr.update(visible=False),
                    )

                bulk_confirm_delete_btn.click(
                    _bulk_confirm_delete,
                    inputs=[ids_state, selected_group, store_tick],
                    outputs=[
                        ids_state,
                        store_tick,
                        selected_group,
                        export_selected_btn,
                        bulk_delete_confirm_panel,
                    ],
                    show_progress="hidden",
                )
                bulk_cancel_delete_btn.click(
                    lambda: gr.update(visible=False),
                    outputs=[bulk_delete_confirm_panel],
                    show_progress="hidden",
                )

                # 进度信息
                def update_progress_info(ids, tick, bump):
                    if not ids:
                        return (
                            gr.update(value="Waiting..."),
                            tick,
                            int(bump or 0),
                        )
                    pending = 0
                    done = 0
                    errors = 0
                    status_signature = []
                    for rid in ids:
                        st = RESULTS_CACHE.get(rid, {})
                        status = st.get("status", "pending")
                        status_signature.append((rid, status))
                        if status == "done":
                            done += 1
                        elif status == "error":
                            errors += 1
                        else:
                            pending += 1
                    qsize = TASK_QUEUE.qsize()
                    running = max(0, pending - qsize)

                    # Info text
                    if pending == 0:
                        info = (
                            f"进度：完成 {done}"
                            + ("" if errors == 0 else f"，错误 {errors}")
                            + "。"
                        )
                    else:
                        info = f"进度：完成 {done}，错误 {errors}，正在解析 {running}，排队 {qsize}，待处理合计 {pending}。"

                    # Only bump render when any item's status changed
                    sig_tuple = tuple(status_signature)
                    last_sig = getattr(update_progress_info, "_last_status_sig", None)
                    bump_out = int(bump or 0)
                    if last_sig != sig_tuple:
                        setattr(update_progress_info, "_last_status_sig", sig_tuple)
                        bump_out = bump_out + 1

                    # Only tick when coarse counts change (avoid unnecessary churn)
                    key = f"{done}_{errors}_{pending}"
                    last_key = getattr(update_progress_info, "_last_counts_key", None)
                    new_tick = int(tick or 0)
                    if last_key != key:
                        setattr(update_progress_info, "_last_counts_key", key)
                        new_tick = new_tick + 1

                    return (
                        gr.update(value=info),
                        new_tick,
                        bump_out,
                    )

                # 计时器不再触达 selected_group，杜绝与用户交互竞争导致选择重置/计时停止
                progress_timer.tick(
                    fn=update_progress_info,
                    inputs=[ids_state, store_tick, render_bump],
                    outputs=[info_display, store_tick, render_bump],
                    show_progress="hidden",
                )

                # Clear all
                def clear_all():
                    global RESULTS_CACHE
                    while not TASK_QUEUE.empty():
                        try:
                            TASK_QUEUE.get_nowait()
                            TASK_QUEUE.task_done()
                        except queue.Empty:
                            break
                    RESULTS_CACHE = {}
                    RETRY_COUNTS.clear()
                    # Do not stop workers; keep them alive
                    return (
                        [],
                        0,
                        gr.update(value="Waiting..."),
                        0,
                        gr.update(choices=[], value=[]),
                        gr.update(value=None),
                    )

                clear_btn.click(
                    clear_all,
                    inputs=None,
                    outputs=[
                        ids_state,
                        store_tick,
                        info_display,
                        render_bump,
                        selected_group,
                        export_selected_btn,
                    ],
                    show_progress="hidden",
                )

    return demo


# ---------------- main ----------------
def _queue_compat(blocks: gr.Blocks):
    """
    Gradio version compatibility layer for Blocks.queue:
    - Try Gradio 4.x: default_concurrency_limit + status_update_rate
    - Fallback to Gradio 3.x: concurrency_count + status_update_rate
    - Final fallback: no-arg queue()
    """
    try:
        # Gradio 4.x path
        return blocks.queue(default_concurrency_limit=20, status_update_rate=0.2)
    except TypeError:
        try:
            # Gradio 3.x path
            return blocks.queue(concurrency_count=16, status_update_rate=0.2)
        except TypeError:
            # Minimal fallback
            return blocks.queue()


def _launch_compat(app: gr.Blocks, port: int):
    """
    Gradio version compatibility for launch parameters.
    """
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            debug=True,
            show_api=False,  # 3.x/部分4.x可用
        )
    except TypeError:
        # Fallback without show_api
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            debug=True,
        )


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    demo = create_gradio_interface()
    app = _queue_compat(demo)
    _launch_compat(app, port)
