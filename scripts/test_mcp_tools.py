# -*- coding: utf-8 -*-
"""
小智本地工具 自动化回归测试脚本
覆盖：
  - Notes：/notes/add /notes/list /notes/delete
  - Reminders：/reminders/parse_create /reminders/list /reminders/cancel /reminders/export/ics
  - 可选探测：/state/* /policy/* /tts/say （不存在则 SKIP）
用法：
  set XIAOZHI_BASE=http://127.0.0.1:8768
  python test_mcp_tools.py
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime

import requests

BASE = os.environ.get("XIAOZHI_BASE", "http://127.0.0.1:8768")
TIMEOUT = 8

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
results: list[dict[str, str]] = []


def rec(test_name: str, status: str, info: str = "") -> None:
    results.append({"name": test_name, "status": status, "info": info})
    tag = {PASS: "✅", FAIL: "❌", SKIP: "⚠️"}[status]
    suffix = f" - {info}" if info else ""
    print(f"{tag} {test_name}{suffix}")


def get(path: str, **kw):
    return requests.get(BASE + path, timeout=TIMEOUT, **kw)


def post(path: str, *, json: dict | None = None, **kw):
    return requests.post(BASE + path, json=json, timeout=TIMEOUT, **kw)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def has_endpoint(method: str, path: str) -> bool:
    try:
        if method.upper() == "GET":
            resp = get(path)
        else:
            resp = post(path)
        return resp.status_code < 500
    except Exception:
        return False


def test_notes_cycle() -> None:
    name = "Notes：新增-查询-删除"
    try:
        text = f"自动化回归测试-笔记-{uuid.uuid4().hex[:8]}"
        resp = post("/notes/add", json={"text": text, "tags": ["autotest"]})
        assert_true(resp.ok, f"/notes/add http {resp.status_code}")
        note_id = resp.json().get("id")
        assert_true(isinstance(note_id, int), "返回 id 非整数")

        resp = get("/notes/list", params={"q": "自动化回归测试"})
        assert_true(resp.ok, f"/notes/list http {resp.status_code}")
        items = resp.json().get("items", [])
        assert_true(any(it.get("id") == note_id for it in items), "新建笔记未出现在查询结果里")

        resp = post("/notes/delete", json={"id": note_id})
        assert_true(resp.ok and resp.json().get("ok"), f"/notes/delete 失败：{resp.text}")

        resp = get("/notes/list", params={"q": "自动化回归测试"})
        assert_true(resp.ok, f"/notes/list http {resp.status_code}")
        items = resp.json().get("items", [])
        assert_true(all(it.get("id") != note_id for it in items), "删除后仍能查询到该笔记")

        rec(name, PASS)
    except Exception as exc:  # pragma: no cover - diagnostic output
        rec(name, FAIL, str(exc))


def test_reminder_once_cancel() -> None:
    name = "Reminders：一次性-创建-列出-取消"
    try:
        marker = f"一次性提醒-{uuid.uuid4().hex[:6]}"
        utterance = f"10分钟后提醒我 {marker}"
        resp = post("/reminders/parse_create", json={"utterance": utterance, "text": marker})
        assert_true(resp.ok, f"/reminders/parse_create http {resp.status_code}")
        payload = resp.json()
        reminder_id = payload.get("id")
        parsed = payload.get("parsed", {})
        assert_true(isinstance(reminder_id, int), "返回 id 非整数")
        assert_true(parsed.get("mode") == "once", f"mode 应为 once，实际 {parsed}")
        ts_utc = parsed.get("ts_utc")
        assert_true(isinstance(ts_utc, (int, float)) and ts_utc > time.time(), "ts_utc 不合法")

        resp = get("/reminders/list", params={"status": "scheduled"})
        assert_true(resp.ok, f"/reminders/list http {resp.status_code}")
        items = resp.json().get("items", [])
        assert_true(any(it.get("id") == reminder_id for it in items), "scheduled 列表未包含刚创建的提醒")

        resp = post("/reminders/cancel", json={"id": reminder_id})
        assert_true(resp.ok and resp.json().get("ok"), f"/reminders/cancel 失败：{resp.text}")

        resp = get("/reminders/list", params={"status": "scheduled"})
        items = resp.json().get("items", [])
        assert_true(all(it.get("id") != reminder_id for it in items), "取消后仍出现在 scheduled 列表")

        rec(name, PASS)
    except Exception as exc:  # pragma: no cover
        rec(name, FAIL, str(exc))


def test_reminder_cron_cancel() -> None:
    name = "Reminders：循环（cron）-创建-取消"
    try:
        marker = f"循环提醒-{uuid.uuid4().hex[:6]}"
        utterance = f"每周一早上9点提醒 {marker}"
        resp = post("/reminders/parse_create", json={"utterance": utterance, "text": marker})
        assert_true(resp.ok, f"/reminders/parse_create http {resp.status_code}")
        payload = resp.json()
        reminder_id = payload.get("id")
        parsed = payload.get("parsed", {})
        assert_true(isinstance(reminder_id, int), "返回 id 非整数")
        assert_true(parsed.get("mode") == "cron", f"mode 应为 cron，实际 {parsed}")
        cron = parsed.get("cron")
        assert_true(isinstance(cron, str) and len(cron.split()) == 5, f"cron 表达不合法：{cron}")

        resp = get("/reminders/list", params={"status": "scheduled"})
        assert_true(resp.ok, f"/reminders/list http {resp.status_code}")
        items = resp.json().get("items", [])
        assert_true(any(it.get("id") == reminder_id for it in items), "scheduled 列表未包含循环提醒")

        resp = post("/reminders/cancel", json={"id": reminder_id})
        assert_true(resp.ok and resp.json().get("ok"), f"/reminders/cancel 失败：{resp.text}")

        rec(name, PASS)
    except Exception as exc:  # pragma: no cover
        rec(name, FAIL, str(exc))


def test_export_ics() -> None:
    name = "Reminders：导出 ICS"
    try:
        resp = get("/reminders/export/ics")
        assert_true(resp.ok, f"/reminders/export/ics http {resp.status_code}")
        body = resp.text
        assert_true("BEGIN:VCALENDAR" in body and "END:VCALENDAR" in body, "ICS 内容不正确")
        rec(name, PASS)
    except Exception as exc:  # pragma: no cover
        rec(name, FAIL, str(exc))


def test_optional_state_policy_tts() -> None:
    if has_endpoint("GET", "/state/get"):
        try:
            resp = get("/state/get")
            assert_true(resp.ok, f"/state/get http {resp.status_code}")
            rec("State：/state/get", PASS)
        except Exception as exc:  # pragma: no cover
            rec("State：/state/get", FAIL, str(exc))
    else:
        rec("State：/state/get", SKIP, "endpoint not found")

    if has_endpoint("GET", "/policy/profile"):
        try:
            resp = get("/policy/profile")
            assert_true(resp.ok, f"/policy/profile http {resp.status_code}")
            if has_endpoint("POST", "/policy/profile"):
                resp_set = post("/policy/profile", json={"name": "home"})
                assert_true(resp_set.ok, f"/policy/profile set http {resp_set.status_code}")
            rec("Policy：profile get/set", PASS)
        except Exception as exc:  # pragma: no cover
            rec("Policy：profile get/set", FAIL, str(exc))
    else:
        rec("Policy：profile get/set", SKIP, "endpoint not found")

    if has_endpoint("POST", "/tts/say"):
        try:
            resp = post("/tts/say", json={"text": "自动化测试：这是一条播报演示"})
            assert_true(resp.ok, f"/tts/say http {resp.status_code}")
            rec("TTS：/tts/say", PASS)
        except Exception as exc:  # pragma: no cover
            rec("TTS：/tts/say", FAIL, str(exc))
    else:
        rec("TTS：/tts/say", SKIP, "endpoint not found")


def main() -> None:
    print(f"== 小智本地工具自检（BASE={BASE}）==")
    try:
        resp = get("/notes/list")
        if not resp.ok:
            print("无法连通服务，请确认 brain_server 正在运行。")
            sys.exit(2)
    except Exception as exc:
        print(f"连通失败：{exc}")
        sys.exit(2)

    test_notes_cycle()
    test_reminder_once_cancel()
    test_reminder_cron_cancel()
    test_export_ics()
    test_optional_state_policy_tts()

    total = len(results)
    ok = sum(1 for item in results if item["status"] == PASS)
    fail = sum(1 for item in results if item["status"] == FAIL)
    skip = sum(1 for item in results if item["status"] == SKIP)

    print("\n=== 测试汇总 ===")
    print(f"TOTAL: {total}  PASS: {ok}  FAIL: {fail}  SKIP: {skip}")

    report = {
        "base": BASE,
        "time": datetime.now().isoformat(),
        "results": results,
        "summary": {"total": total, "pass": ok, "fail": fail, "skip": skip},
    }
    with open("test_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open("test_report.txt", "w", encoding="utf-8") as fh:
        for item in results:
            info = f" {item['info']}" if item.get("info") else ""
            fh.write(f"{item['status']:4}  {item['name']}{info}\n")
        fh.write(f"\nTOTAL: {total}  PASS: {ok}  FAIL: {fail}  SKIP: {skip}\n")

    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
