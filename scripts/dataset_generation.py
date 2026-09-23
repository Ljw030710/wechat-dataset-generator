#!/usr/bin/env python3
"""Shared model clients and validation for fixed-format chat datasets.

The private and group generators import this shared infrastructure module.
Keep the JSON contract here strict: downstream screenshot tools rely on these
exact top-level keys and message/schedule field names.
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import difflib
import http.client
import json
import os
import pathlib
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any
from typing import Literal

from dotenv import dotenv_values

from schedule_evidence import SCHEDULE_GROUNDING_RULES
from schedule_evidence import validate_schedule_evidence

ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# CLIProxyAPI 是可选提供商；未指定提供商且未配置 LLM_PROVIDER 时默认 DeepSeek。
# 优先读取 CLIPROXY_API_KEY，未设置时自动读取 Homebrew 配置中的第一个 api-keys 条目。
CLIPROXY_API_URL = "http://127.0.0.1:8317/v1/chat/completions"
# 2026-08-24 时本机 CLIProxyAPI 的可用模型之一。可通过 --model 覆盖，或
# 在环境变量或 .env 中设置 CLIPROXY_MODEL；--list-models 可查看实时清单。
CLIPROXY_MODEL = os.environ.get("CLIPROXY_MODEL", "gemini-3.7-flash-high")
CLIPROXY_CONFIG_PATH = pathlib.Path("/opt/homebrew/etc/cliproxyapi.conf")

PRIVATE_TOP_LEVEL_KEYS = (
    "conversation_id",
    "messages",
    "participants",
    "schedule",
    "topic",
)
GROUP_TOP_LEVEL_KEYS = PRIVATE_TOP_LEVEL_KEYS + ("group_name",)
MESSAGE_KEYS = ("speaker", "text", "time")
SCHEDULE_KEYS = ("date", "owner", "task", "time")
ChatType = Literal["private", "group"]
MAX_GENERATION_ATTEMPTS = 20
DOMAIN_RECENCY_WINDOW = 2
GENERIC_ENDING_KEYS = {
    "到时见",
    "回头见",
    "明天见",
    "那就这样",
    "就这么定了",
    "稳了",
    "这下稳了",
    "那就稳了",
    "好嘞",
    "好呀",
    "好的",
    "收到",
}
NAME_PREFIXES = ("小", "老", "阿")
NAME_SUFFIXES = (
    "老师",
    "同学",
    "经理",
    "主任",
    "老板",
    "师傅",
    "哥",
    "姐",
    "叔",
    "姨",
    "总",
)
TOPIC_DOMAIN_PATTERNS = (
    (
        "图书出版",
        r"书店|图书馆|绘本|诗集|古籍|手稿|书评|出版|编辑|主编|稿件|封面|作者|读书|旧书|新书|原版书|书单|荐书|装帧|摄影集|诗人|读书会|读书月",
    ),
    (
        "园艺种植",
        r"花市|花苗|花盆|绿萝|吊兰|月季|绣球|园艺|菜园|种花|堆肥|蚯蚓土|果农|番茄苗|菜友|石榴树|养花",
    ),
    (
        "餐饮烹饪",
        r"做饭|做菜|烘焙|蛋糕|曲奇|菜谱|食谱|餐厅|火锅|烧烤|外卖|买菜",
    ),
    ("休闲探店", r"探店|咖啡馆|咖啡店|美术馆|艺术展|展馆|逛展"),
    (
        "出游户外",
        r"旅行|旅游|露营|爬山|徒步|景区|酒店|民宿|机票|车票|野餐|山野",
    ),
    (
        "学习教育",
        r"作业|考试|错题|课程|课堂|学生|老师|作文|语文|英语|英文|高中生|家长会|论文|答辩|培训|补课|复习|自习室|占座",
    ),
    (
        "职场协作",
        r"客户|同事|会议|项目|需求|版本|上线|汇报|合同|报表|排班|加班",
    ),
    ("家庭育儿", r"孩子|宝宝|幼儿园|接娃|育儿|亲子|老人|爸妈|家务|家庭聚会"),
    (
        "医疗健康",
        r"医院|护士|医生|病人|输液|查房|急诊|产科|陪护|待产|复诊|体检|药店|康复|看病",
    ),
    ("宠物动物", r"猫|狗|宠物|兽医|猫粮|狗粮|遛狗|领养"),
    (
        "运动健身",
        r"健身|跑步|晨跑|夜跑|心率|恢复咨询|羽毛球|篮球|足球|游泳|瑜伽|骑行|训练",
    ),
    (
        "社区居住",
        r"小区|社区|物业|楼道|楼顶|邻居|街坊|停车位|业委会|租房|搬家|搬运|维修|修复|沙发|衣柜|小院|老院|修补",
    ),
    ("数码技术", r"手机|电脑|软件|程序|系统|账号|网络|相机|数码|游戏"),
    (
        "文娱创作",
        r"电影|演出|展览|音乐|摄影|画展|古画|字画|鉴定|收藏|陶艺|插画|手帐|画画|剧本|排练|乐队|手作",
    ),
)
RETRY_DOMAIN_HINTS = tuple(
    label for label, _pattern in TOPIC_DOMAIN_PATTERNS if label != "医疗健康"
)


def compact_json(value: Any) -> str:
    """Encode compact JSON while keeping Chinese characters readable."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def topic_domain_tags(topic: str, group_name: str = "") -> set[str]:
    """Classify topics for diversity checks, not as generation templates."""
    text = f"{group_name} {topic}"
    return {
        label
        for label, pattern in TOPIC_DOMAIN_PATTERNS
        if re.search(pattern, text, re.I)
    }


def conversation_domain_tags(value: dict[str, Any]) -> set[str]:
    """Use message content to catch domains hidden by vague topics."""
    parts = [value.get("topic", ""), value.get("group_name", "")]
    messages = value.get("messages")
    if isinstance(messages, list):
        parts.extend(
            message.get("text", "")
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("text"), str)
        )
    return topic_domain_tags(" ".join(parts))


def conversation_primary_domain_tags(value: dict[str, Any]) -> set[str]:
    """Prefer topic and group-name domains, falling back to message content."""
    topic = (
        value.get("topic", "") if isinstance(value.get("topic"), str) else ""
    )
    group_name = (
        value.get("group_name", "")
        if isinstance(value.get("group_name"), str)
        else ""
    )
    explicit = topic_domain_tags(topic, group_name)
    return explicit or conversation_domain_tags(value)


def reject_forbidden_brief_domains(brief_text: str, direction: str) -> None:
    """Reject a brief that repeats a recent domain before generating a draft."""
    match = re.search(
        r"本地分类器判定最近 \d+ 条已覆盖这些领域：(\[[^\n]*?\])。",
        direction,
    )
    if not match:
        return
    try:
        forbidden = set(json.loads(match.group(1)))
    except (json.JSONDecodeError, TypeError):
        return
    overlap = topic_domain_tags(brief_text) & forbidden
    if overlap:
        raise ValueError(
            f"蓝图生活领域已与最近数据重复（{','.join(sorted(overlap))}），无需继续写草稿"
        )


def reject_forbidden_brief_schedule(brief_text: str, direction: str) -> None:
    """Reject planning briefs when the direction forbids more schedules."""
    if "最终 schedule 必须为 []" not in direction:
        return
    if re.search(
        r"约见|见面|碰头|聚会|改期|改时间|约定|邀约|一起去|下次去|周末去",
        brief_text,
    ):
        raise ValueError(
            "最近日程密度已满，蓝图仍在规划约见或待办，无需继续写草稿"
        )


def extract_json_object(raw: str) -> dict[str, Any]:
    """Accept a plain JSON response or a fenced JSON response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型回复中没有 JSON 对象")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型回复必须是 JSON 对象")
    return value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_formal_chinese_name(value: Any) -> bool:
    """Accept 2–3 Han-character names, excluding nicknames and titles."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[\u3400-\u4DBF\u4E00-\u9FFF]{2,3}", value
    ):
        return False
    return not value.startswith(NAME_PREFIXES) and not value.endswith(
        NAME_SUFFIXES
    )


def validate_conversation(value: Any, chat_type: ChatType) -> dict[str, Any]:
    """Validate the fixed conversation contract without modifying the input.

    Args:
        value: Candidate JSON object to validate.
        chat_type: Private or group schema, including participant count rules.

    Returns:
        The original dictionary with its field order and values unchanged.

    Raises:
        ValueError: Fields, names, message order, lengths, or dates are invalid.
    """
    if not isinstance(value, dict):
        raise ValueError("会话必须是 JSON 对象")
    top_level_keys = (
        PRIVATE_TOP_LEVEL_KEYS
        if chat_type == "private"
        else GROUP_TOP_LEVEL_KEYS
    )
    missing = [key for key in top_level_keys if key not in value]
    extra = [key for key in value if key not in top_level_keys]
    if missing or extra:
        raise ValueError(f"顶层字段不固定；缺少={missing}，多出={extra}")
    if not _nonempty_string(value["conversation_id"]) or not _nonempty_string(
        value["topic"]
    ):
        raise ValueError("conversation_id 和 topic 必须是非空字符串")
    if chat_type == "group":
        group_name = value["group_name"]
        if (
            not _nonempty_string(group_name)
            or not 2 <= len(group_name.strip()) <= 12
        ):
            raise ValueError("group_name 必须是 2–12 个字符的正常群聊名称")

    participants = value["participants"]
    expected = 2 if chat_type == "private" else None
    if not isinstance(participants, list) or not all(
        _nonempty_string(item) for item in participants
    ):
        raise ValueError("participants 必须是非空字符串数组")
    if expected and len(participants) != expected:
        raise ValueError("私聊 participants 必须恰好有 2 人")
    if chat_type == "group" and not 3 <= len(participants) <= 5:
        raise ValueError("群聊 participants 必须有 3–5 人")
    participant_names = [
        re.split(r"[（(]", item, maxsplit=1)[0].strip() for item in participants
    ]
    invalid_names = [
        name for name in participant_names if not is_formal_chinese_name(name)
    ]
    if invalid_names:
        raise ValueError(
            f"参与者必须使用 2–3 个汉字的正式姓名：{invalid_names}"
        )
    if len(set(participant_names)) != len(participant_names):
        raise ValueError("participants 中的姓名不能重复")

    messages = value["messages"]
    if not isinstance(messages, list) or not 10 <= len(messages) <= 12:
        raise ValueError("messages 必须有 10–12 条")
    previous_time: datetime.datetime | None = None
    used_speakers: set[str] = set()
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or tuple(message.keys()) != MESSAGE_KEYS
        ):
            raise ValueError(
                f"messages[{index}] 字段及顺序必须为 {MESSAGE_KEYS}"
            )
        if not all(_nonempty_string(message[key]) for key in MESSAGE_KEYS):
            raise ValueError(f"messages[{index}] 的字段不能为空")
        if message["speaker"] not in participant_names:
            raise ValueError(f"messages[{index}].speaker 不在 participants 中")
        used_speakers.add(message["speaker"])
        try:
            parsed = datetime.datetime.strptime(
                message["time"], "%Y-%m-%d %H:%M"
            )
        except ValueError as error:
            raise ValueError(
                f"messages[{index}].time 必须为 YYYY-MM-DD HH:MM"
            ) from error
        if previous_time and parsed < previous_time:
            raise ValueError("消息时间必须按先后顺序排列")
        previous_time = parsed
        if len(message["text"]) > 28:
            raise ValueError(f"messages[{index}].text 超过 28 字")
    required_speakers = 2 if chat_type == "private" else 3
    if len(used_speakers) < required_speakers:
        raise ValueError(f"对话实际发言人数不能少于 {required_speakers} 人")
    schedule = value["schedule"]
    if not isinstance(schedule, list) or len(schedule) > 3:
        raise ValueError("schedule 必须是数组，最多 3 项")
    for index, item in enumerate(schedule):
        if not isinstance(item, dict) or tuple(item.keys()) != SCHEDULE_KEYS:
            raise ValueError(
                f"schedule[{index}] 字段及顺序必须为 {SCHEDULE_KEYS}"
            )
        if not all(_nonempty_string(item[key]) for key in SCHEDULE_KEYS):
            raise ValueError(f"schedule[{index}] 的字段不能为空")
        if item["owner"] not in participant_names:
            raise ValueError(f"schedule[{index}].owner 不在 participants 中")
        try:
            datetime.datetime.strptime(item["date"], "%Y-%m-%d")
            datetime.datetime.strptime(item["time"], "%H:%M")
        except ValueError as error:
            raise ValueError(
                f"schedule[{index}] 日期或时间格式错误：date={item['date']!r}, time={item['time']!r}"
            ) from error

    return value


def read_model_settings() -> dict[str, str]:
    """Read project-local settings without changing the process environment."""
    values = (
        dotenv_values(ENV_FILE, interpolate=False) if ENV_FILE.is_file() else {}
    )
    return {
        **{key: value for key, value in values.items() if value is not None},
        **os.environ,
    }


def load_cliproxy_api_key() -> str:
    """Load the local proxy key without hard-coding it into this source file."""
    environment_key = read_model_settings().get("CLIPROXY_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        config_lines = CLIPROXY_CONFIG_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as error:
        raise ValueError(
            "无法读取 CLIProxyAPI API Key；请设置 CLIPROXY_API_KEY 环境变量，"
            f"或检查配置文件 {CLIPROXY_CONFIG_PATH}"
        ) from error

    inside_api_keys = False
    for raw_line in config_lines:
        stripped = raw_line.strip()
        if stripped == "api-keys:":
            inside_api_keys = True
            continue
        if not inside_api_keys:
            continue
        if raw_line and not raw_line[0].isspace():
            break
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            api_key = stripped[1:].strip().strip('"').strip("'")
            if api_key:
                return api_key

    raise ValueError(
        f"{CLIPROXY_CONFIG_PATH} 中没有可用的 api-keys；"
        "请在管理页面添加，或设置 CLIPROXY_API_KEY 环境变量"
    )


class ChatCompletionClient:
    """A JSON chat-completion client with bounded retries.

    Attributes:
        api_key: Credential sent in the Authorization header.
        api_url: Chat-completion endpoint for the selected provider.
        model: Model ID included in each request.
        provider_name: Provider label used in error messages.
        timeout: Timeout in seconds for each HTTP request.
        request_attempts: Maximum attempts per completion, including the first.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 180,
        request_attempts: int = 3,
        *,
        api_url: str = DEEPSEEK_API_URL,
        model: str = DEEPSEEK_MODEL,
        provider_name: str = "DeepSeek",
    ) -> None:
        """Configure the client, reading omitted DeepSeek keys from local settings.

        Credentials are resolved at construction time so importing this module
        does not require a key. An explicit key takes precedence, including an
        empty value, which is rejected rather than replaced with another key.
        """
        if api_key is None and provider_name == "DeepSeek":
            api_key = read_model_settings().get("DEEPSEEK_API_KEY", "")
        api_key = api_key or ""
        api_key = api_key.strip()
        if not api_key or "替换" in api_key:
            if provider_name == "DeepSeek":
                raise ValueError(
                    "请先设置 DEEPSEEK_API_KEY 环境变量，或在项目 .env 中填写"
                )
            raise ValueError(f"请先配置 {provider_name} API Key")
        if provider_name == "DeepSeek" and not api_key.startswith("sk-"):
            raise ValueError(
                "DEEPSEEK_API_KEY 格式无效，请提供以 sk- 开头的密钥"
            )
        if request_attempts < 1:
            raise ValueError("request_attempts 必须至少为 1")
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.provider_name = provider_name
        self.timeout = timeout
        self.request_attempts = request_attempts

    def complete(
        self, system: str, user: str, *, temperature: float
    ) -> dict[str, Any]:
        """Request one JSON object, retrying transient failures.

        Args:
            system: Instructions defining the model's role and output contract.
            user: Input for the current prompt-chain stage.
            temperature: Sampling temperature sent to the provider.

        Returns:
            The parsed JSON object from the first completion choice.

        Raises:
            RuntimeError: A non-retryable HTTP error occurs or attempts run out.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        last_error: RuntimeError | None = None
        for attempt in range(1, self.request_attempts + 1):
            request = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return extract_json_object(
                    body["choices"][0]["message"]["content"]
                )
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"{self.provider_name} API 返回 HTTP {error.code}: {detail[:500]}"
                )
                if error.code not in (408, 409, 429) and error.code < 500:
                    raise last_error from error
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                ssl.SSLError,
            ) as error:
                last_error = RuntimeError(
                    f"无法连接 {self.provider_name} API：{error}"
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                last_error = RuntimeError(
                    f"{self.provider_name} API 返回的 JSON 结构异常：{error}"
                )

            if attempt < self.request_attempts:
                print(
                    f"  API 请求第 {attempt} 次失败，正在重试：{last_error}",
                    file=sys.stderr,
                )
                time.sleep(0.5 * attempt)

        raise last_error or RuntimeError(f"{self.provider_name} API 请求失败")


def list_cliproxy_models(timeout: int = 15) -> list[str]:
    """Return the model IDs currently exposed by the local CLIProxyAPI."""
    models_url = CLIPROXY_API_URL.rsplit("/", 2)[0] + "/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {load_cliproxy_api_key()}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"CLIProxyAPI 模型列表返回 HTTP {error.code}: {detail[:500]}"
        ) from error
    except (
        urllib.error.URLError,
        TimeoutError,
        ssl.SSLError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(f"无法读取 CLIProxyAPI 模型列表：{error}") from error

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("CLIProxyAPI 模型列表格式异常：缺少 data 数组")
    model_ids = [
        item["id"]
        for item in data
        if isinstance(item, dict) and _nonempty_string(item.get("id"))
    ]
    if not model_ids:
        raise RuntimeError("CLIProxyAPI 当前没有暴露可用模型")
    return model_ids


def build_generation_client(
    provider: str, *, model: str | None = None, base_url: str | None = None
) -> ChatCompletionClient:
    """Build a generation or teacher client with provider-specific credentials."""
    settings = read_model_settings()
    if base_url is not None and provider != "custom":
        raise ValueError("--base-url 仅适用于 --provider custom")
    if provider == "deepseek":
        return ChatCompletionClient(
            model=model or settings.get("DEEPSEEK_MODEL") or DEEPSEEK_MODEL
        )
    if provider in ("cliproxy", "sol"):
        return ChatCompletionClient(
            load_cliproxy_api_key(),
            api_url=CLIPROXY_API_URL,
            model=model or settings.get("CLIPROXY_MODEL") or CLIPROXY_MODEL,
            provider_name="CLIProxyAPI",
        )
    if provider == "custom":
        custom_key = settings.get("LLM_API_KEY", "").strip()
        if not custom_key:
            raise ValueError(
                "请先设置 LLM_API_KEY 环境变量，或在项目 .env 中填写"
            )
        custom_url = (
            (
                base_url
                if base_url is not None
                else settings.get("LLM_BASE_URL", "")
            )
            .strip()
            .rstrip("/")
        )
        custom_model = (
            model if model is not None else settings.get("LLM_MODEL", "")
        ).strip()
        if not custom_url:
            raise ValueError("请设置 LLM_BASE_URL 或传入 --base-url")
        if not custom_model:
            raise ValueError("请设置 LLM_MODEL 或传入 --model")
        try:
            parsed = urllib.parse.urlsplit(custom_url)
            valid_url = (
                parsed.scheme in ("http", "https")
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and not any(char.isspace() for char in custom_url)
            )
            parsed.port  # Validate malformed port numbers before any request.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(
                "自定义接口必须是 HTTP(S) 地址，不能包含凭据、查询参数或片段"
            )
        if not custom_url.endswith("/chat/completions"):
            custom_url += "/chat/completions"
        return ChatCompletionClient(
            custom_key,
            api_url=custom_url,
            model=custom_model,
            provider_name="Custom",
        )
    raise ValueError(f"不支持的模型提供商：{provider}")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Share provider options across generation and teacher-export commands."""
    parser.add_argument(
        "--provider",
        choices=("deepseek", "cliproxy", "sol", "custom"),
        default=read_model_settings().get("LLM_PROVIDER", "deepseek"),
        help="默认读取 LLM_PROVIDER，未配置时为 deepseek；sol 为 cliproxy 的兼容别名",
    )
    parser.add_argument(
        "--model",
        help="模型 ID；custom 默认读取 LLM_MODEL",
    )
    parser.add_argument(
        "--base-url",
        help="custom 接口基础地址（含版本路径），默认读取 LLM_BASE_URL",
    )


def schema_instruction(chat_type: ChatType) -> str:
    """Return the fixed JSON schema instructions for the selected chat type."""
    count = "恰好2人" if chat_type == "private" else "3至5人"
    top_level = "conversation_id, messages, participants, schedule, topic"
    group_name_rule = ""
    if chat_type == "group":
        top_level += ", group_name"
        group_name_rule = "\ngroup_name 为2–12个字符的自然群名，不是事件摘要；topic 才是内容摘要。"
    return f"""输出只能是一个 JSON 对象，顶层字段和顺序固定为：
{top_level}。
messages 每项字段和顺序固定为 speaker, text, time；共 10–12 条；每条不超过28个汉字；尽量控制在8–22字；time 格式 YYYY-MM-DD HH:MM。
participants 为{count}，格式为“姓名（简短角色特点）”。姓名必须是正式中文姓名，只能由 2–3 个汉字组成；禁止昵称、称谓、英文名、拼音、数字或“用户A”之类代号。
messages.speaker 必须逐字使用 participants 括号前的正式姓名，不能使用昵称或称谓。
schedule 每项字段和顺序固定为 date, owner, task, time；只保留最终仍有效的安排，0–3项；owner 也必须逐字使用 participants 中的正式姓名。
不得增加 chat_type、title、summary 等任何字段。{group_name_rule}
{SCHEDULE_GROUNDING_RULES}"""


def validate_with_repair(
    client: ChatCompletionClient,
    value: dict[str, Any],
    chat_type: ChatType,
    *,
    max_repairs: int = 2,
    group_plan: dict[str, Any] | None = None,
    private_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a draft, then ask the model to repair validation failures.

    Args:
        client: Client used when local normalization cannot fix a draft.
        value: Draft to validate; local normalizers modify it in place.
        chat_type: Conversation schema to enforce.
        max_repairs: Maximum number of model-assisted repairs after the first
            validation attempt.
        group_plan: Optional accepted group roster and message plan.
        private_plan: Optional accepted private roster and message plan.

    Returns:
        A validated conversation with the original conversation ID preserved.
        A model repair may replace the input object.

    Raises:
        ValueError: The draft still fails validation after all repairs.
        RuntimeError: A model request fails after its own retries.
    """
    candidate = value
    conversation_id = value.get("conversation_id")
    last_error: ValueError | None = None
    for repair_index in range(max_repairs + 1):
        candidate = normalize_schedule_datetimes(candidate)
        candidate = normalize_message_markers(candidate)
        candidate = normalize_simulation_topic(candidate)
        candidate = shorten_message_texts(candidate)
        if chat_type == "private" and private_plan is not None:
            candidate = apply_private_plan_constraints(candidate, private_plan)
        elif chat_type == "group":
            candidate = (
                apply_group_plan_constraints(candidate, group_plan)
                if group_plan is not None
                else rebuild_group_participants(candidate)
            )
        if group_plan is not None or private_plan is not None:
            candidate = normalize_calendar_mentions(candidate)
        try:
            validated = validate_conversation(candidate, chat_type)
            if group_plan is not None or private_plan is not None:
                validate_calendar_mentions(validated)
                validate_natural_dialogue(validated)
                validate_human_dialogue_style(validated)
            if chat_type == "private" and private_plan is not None:
                validate_private_training_schedule(validated)
            if chat_type == "group" and group_plan is not None:
                validate_group_training_schedule(validated)
                validate_closed_group_ending(validated)
            if group_plan is not None or private_plan is not None:
                validate_schedule_evidence(client, validated)
            return validated
        except ValueError as error:
            last_error = error
            if repair_index >= max_repairs:
                break
            candidate = client.complete(
                "你是 JSON 与自然口语修复器。只修复明确的校验错误，不改变人物、剧情和有效安排。",
                f"""待修复会话：{compact_json(candidate)}
校验错误：{error}

若错误为标签缺少正文依据，优先从 task 删除正文未支持的附加细节，不得把隐含背景补写进台词来通过检查。
不能删去核心活动来掩盖错误；若核心约定没有成立，应修复约定表达并重新核验，无法修复时本条会被拒绝。

	请只修复导致错误的字段。若短消息过多形成电报体，把其中一部分补成 8–18 字、带具体上下文的
	自然消息；若出现会议纪要式套话，改成符合人物关系的口语。保持消息数量、speaker 顺序和事实
	不变，不要新增“哈哈”来假装自然。
所有参与者必须使用 2–3 个汉字的正式中文姓名；
speaker 和 schedule.owner 必须逐字复制 participants 括号前姓名，禁止昵称、称谓或代号。
群聊不得在 participants 之外临时新增发言者。
schedule 中日期必须是真实存在的 YYYY-MM-DD，时间必须是 HH:MM；
schedule 只保留聊天结束时仍有效的未来安排。消息文字不得超过 28 个字符。
私聊明确约定训练样本必须有且仅有 1 项 schedule，owner 必须是 participants[0]；
该绿色发送者的消息正文必须直接出现最终任务、M月D日和无歧义的具体钟点；消息里优先写
“晚上七点半”“下午3点”这类自然时间，禁止把“M月D日HH:MM”粘成日历字段。
不能只靠白色消息补全，也不要让绿色方照抄 schedule.task 的书面标题。
群聊明确约定训练样本使用相同规则：唯一 schedule 必须属于 participants[0]，并能从其绿色气泡直接读取。
输出完整 JSON 对象，不要解释。

{schema_instruction(chat_type)}""",
                temperature=0.1,
            )
            if isinstance(conversation_id, str):
                candidate["conversation_id"] = conversation_id
    raise ValueError(f"最终 JSON 修复 {max_repairs} 次后仍不合格：{last_error}")


def shorten_message_texts(value: dict[str, Any]) -> dict[str, Any]:
    """Shorten overlong bubbles in place, preserving complete clauses."""
    messages = value.get("messages")
    if not isinstance(messages, list):
        return value
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("text"), str
        ):
            continue
        text = message["text"].strip()
        if len(text) <= 28:
            continue
        segments = [
            part.strip()
            for part in re.findall(r"[^。！？；，,!?;]+[。！？；，,!?;]?", text)
        ]
        shortened = ""
        for segment in segments:
            if len(shortened + segment) > 28:
                break
            shortened += segment
        shortened = shortened.rstrip("，,；;")
        if len(shortened) >= 6:
            if shortened[-1] not in "。！？!?" and len(shortened) < 28:
                shortened += "。"
            message["text"] = shortened
    return value


def normalize_schedule_datetimes(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize schedule dates in place, clamping impossible month-end days."""

    def parse_chinese_number(text: str) -> int | None:
        if text.isdigit():
            return int(text)
        digits = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if text == "十":
            return 10
        if "十" in text:
            left, right = text.split("十", maxsplit=1)
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        if all(char in digits for char in text):
            return int("".join(str(digits[char]) for char in text))
        return None

    schedule = value.get("schedule")
    if not isinstance(schedule, list):
        return value
    for item in schedule:
        if not isinstance(item, dict):
            continue
        date_value = item.get("date")
        if isinstance(date_value, str):
            match = re.fullmatch(
                r"\s*(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\s*", date_value
            )
            if match:
                year, month, day = map(int, match.groups())
                if 1 <= year <= 9999 and 1 <= month <= 12:
                    day = min(max(day, 1), calendar.monthrange(year, month)[1])
                    item["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        time_value = item.get("time")
        if isinstance(time_value, str):
            match = re.fullmatch(r"\s*(\d{1,2})[:：](\d{1,2})\s*", time_value)
            if match:
                hour, minute = map(int, match.groups())
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    item["time"] = f"{hour:02d}:{minute:02d}"
                    continue
            match = re.fullmatch(
                r"\s*(上午|下午|晚上|中午|凌晨)?\s*(\d{1,2})(?:点|时)(半|\d{1,2}分?)?\s*",
                time_value,
            )
            if not match:
                match = re.fullmatch(
                    r"\s*(上午|下午|晚上|中午|凌晨)?\s*"
                    r"([零〇一二两三四五六七八九十]+)(?:点|时)"
                    r"(半|一刻|三刻|[零〇一二两三四五六七八九十]+分?)?\s*",
                    time_value,
                )
            if match:
                period, hour_text, minute_text = match.groups()
                hour = parse_chinese_number(hour_text)
                if minute_text == "半":
                    minute = 30
                elif minute_text == "一刻":
                    minute = 15
                elif minute_text == "三刻":
                    minute = 45
                else:
                    minute = parse_chinese_number(
                        (minute_text or "0").rstrip("分")
                    )
                if hour is None or minute is None:
                    continue
                if period in ("下午", "晚上") and 1 <= hour < 12:
                    hour += 12
                elif period == "中午" and 1 <= hour < 11:
                    hour += 12
                elif period == "凌晨" and hour == 12:
                    hour = 0
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    item["time"] = f"{hour:02d}:{minute:02d}"
    return value


def normalize_message_markers(value: dict[str, Any]) -> dict[str, Any]:
    """Strip planning-only markers from messages in place and return value."""
    messages = value.get("messages")
    if not isinstance(messages, list):
        return value
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("text"), str
        ):
            continue
        message["text"] = re.sub(
            r"^(?:\[|【)(?:文字|文本)(?:\]|】)\s*", "", message["text"]
        ).strip()
    return value


def normalize_simulation_topic(value: dict[str, Any]) -> dict[str, Any]:
    """Remove irrelevant simulation labels from the topic in place."""
    topic = value.get("topic")
    if not isinstance(topic, str) or "模拟案例" not in topic:
        return value
    service_terms = r"服务评价|顾客反馈|客户评价|用户评价|消费体验|使用体验|商家|客服|售后|证言"
    if not re.search(service_terms, topic):
        topic = re.sub(r"[（(]?模拟案例[）)]?[：:]?", "", topic)
        value["topic"] = topic.strip(" ：:，,（）()")
    return value


def validate_closed_group_ending(value: dict[str, Any]) -> None:
    """Reject group endings that reopen a settled decision."""
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    final_text = (
        messages[-1].get("text", "") if isinstance(messages[-1], dict) else ""
    )
    if final_text.endswith(("？", "?")) or re.search(
        r"建议|要不|大家觉得|行吗|可以吗", final_text
    ):
        raise ValueError(
            "群聊最后一条必须确认最终结果，不能继续提问或提出新变更"
        )


def validate_natural_dialogue(value: dict[str, Any]) -> None:
    """Check dialogue traits that prompt instructions alone cannot guarantee."""
    messages = value.get("messages")
    if not isinstance(messages, list):
        return
    texts = [
        message.get("text", "").strip()
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("text"), str)
    ]
    content_lengths = [
        len(
            re.sub(
                r"[\s，。！？!?、；;：:…~～,.\"'“”‘’（）()\[\]【】]+", "", text
            )
        )
        for text in texts
    ]
    short_count = sum(2 <= length <= 7 for length in content_lengths)
    if short_count > 6:
        raise ValueError(
            f"活人感失衡：2–7 字的短消息最多保留 6 条，当前有 {short_count} 条，像电报体"
        )
    schedule = value.get("schedule")
    if isinstance(schedule, list) and len(schedule) > 1:
        raise ValueError(
            f"自然聊天的 schedule 最多保留 1 项，当前有 {len(schedule)} 项，像任务清单"
        )
    robotic_patterns = (
        r"最终(?:确定|确认)",
        r"方案通过",
        r"^分工明确[。！!]?$",
        r"大家一致同意",
        r"愉快决定",
        r"^确认(?:[，,。]|$)",
        r"相信.{1,8}(?:潜力|能力)",
        r"等着看惊喜",
    )
    hits = [
        text
        for text in texts
        if any(re.search(pattern, text) for pattern in robotic_patterns)
    ]
    if hits:
        raise ValueError(f"存在会议纪要式或模型式表达：{hits[:2]}")


def validate_human_dialogue_style(value: dict[str, Any]) -> None:
    """Reject stilted calendar wording and overly terse dialogue."""
    messages = value.get("messages")
    if not isinstance(messages, list):
        return
    texts = [
        message.get("text", "")
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("text"), str)
    ]
    glued = [
        text
        for text in texts
        if re.search(r"\d{1,2}月\d{1,2}[日号]\s*\d{1,2}:\d{2}", text)
    ]
    if glued:
        raise ValueError(
            f"消息像日历记录：日期与 HH:MM 不得直接粘连，请改用自然中文钟点：{glued[:1]}"
        )

    schedule = value.get("schedule")
    if (
        not isinstance(schedule, list)
        or not schedule
        or not isinstance(schedule[0], dict)
    ):
        return
    task = re.sub(r"[\s，。！？!?、；;：:]+", "", schedule[0].get("task", ""))
    dialogue = re.sub(r"[\s，。！？!?、；;：:]+", "", " ".join(texts))
    if len(task) >= 12 and task in dialogue:
        raise ValueError(
            "绿色消息不得照抄较长的 schedule.task 标题，请用当事人的短动作自然表达"
        )


def _chinese_clock_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    return (
        digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    )


def green_text_has_explicit_time(text: str, time_value: str) -> bool:
    """Match explicit clock wording without losing AM/PM certainty."""
    try:
        hour, minute = map(int, time_value.split(":"))
    except (AttributeError, TypeError, ValueError):
        return False
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return False

    compact = re.sub(r"\s+", "", text)
    direct_tokens = {time_value, f"{hour}:{minute:02d}"}

    def clock_tokens(clock_hour: int, *, include_bare: bool) -> set[str]:
        hour_forms = {str(clock_hour), _chinese_clock_number(clock_hour)}
        if minute == 0:
            suffixes = {"点", "点整"}
        elif minute == 15:
            suffixes = {"点15分", "点一刻"}
        elif minute == 30:
            suffixes = {"点30分", "点半"}
        elif minute == 45:
            suffixes = {"点45分", "点三刻"}
        else:
            suffixes = {
                f"点{minute}",
                f"点{minute}分",
                f"点{_chinese_clock_number(minute)}",
                f"点{_chinese_clock_number(minute)}分",
            }
        tokens = {
            hour_text + suffix
            for hour_text in hour_forms
            for suffix in suffixes
        }
        return tokens if include_bare else set()

    # A 24-hour Chinese clock such as “19点30分” is already unambiguous.
    if hour > 12:
        direct_tokens.update(clock_tokens(hour, include_bare=True))

    twelve_hour = hour % 12 or 12
    if 0 <= hour < 6:
        periods = ("凌晨",)
    elif 6 <= hour < 9:
        periods = ("早上", "上午")
    elif 9 <= hour < 12:
        periods = ("上午",)
    elif 12 <= hour < 14:
        periods = ("中午", "下午")
    elif 14 <= hour < 18:
        periods = ("下午",)
    elif hour < 20:
        periods = ("傍晚", "晚上")
    else:
        periods = ("晚上",)
    natural_clock_tokens = clock_tokens(twelve_hour, include_bare=True)
    direct_tokens.update(
        period + token for period in periods for token in natural_clock_tokens
    )
    return any(token in compact for token in direct_tokens)


def validate_private_training_schedule(value: dict[str, Any]) -> None:
    """Require one future agreement stated explicitly in green bubbles."""
    participants = value.get("participants")
    messages = value.get("messages")
    schedule = value.get("schedule")
    if not isinstance(participants, list) or len(participants) != 2:
        raise ValueError("私聊明确约定校验需要恰好 2 名参与者")
    if not isinstance(messages, list) or not messages:
        raise ValueError("私聊明确约定校验需要有效消息")
    if not isinstance(schedule, list) or len(schedule) != 1:
        raise ValueError(
            "私聊训练样本必须有且仅有 1 项明确未来约定，schedule 不能是空数组"
        )

    green_name = re.split(r"[（(]", participants[0], maxsplit=1)[0].strip()
    item = schedule[0]
    if not isinstance(item, dict) or item.get("owner") != green_name:
        raise ValueError(
            "schedule.owner 必须是 participants[0]（右侧绿色气泡发送者）"
        )

    try:
        scheduled_at = datetime.datetime.strptime(
            f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M"
        )
        latest_message_at = max(
            datetime.datetime.strptime(message["time"], "%Y-%m-%d %H:%M")
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("time"), str)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("无法核对私聊约定与消息时间") from error
    if scheduled_at <= latest_message_at:
        raise ValueError("schedule 必须是晚于聊天结束时间的未来约定")

    green_text = " ".join(
        message.get("text", "")
        for message in messages
        if isinstance(message, dict)
        and message.get("speaker") == green_name
        and isinstance(message.get("text"), str)
    )
    schedule_date = datetime.datetime.strptime(item["date"], "%Y-%m-%d")
    date_tokens = (
        item["date"],
        f"{schedule_date.year}年{schedule_date.month}月{schedule_date.day}日",
        f"{schedule_date.month}月{schedule_date.day}日",
        f"{schedule_date.month}月{schedule_date.day}号",
    )
    if not any(token in green_text for token in date_tokens):
        raise ValueError(
            "绿色气泡必须直接写出最终约定的 M月D日，不能只靠白色气泡或相对代称"
        )
    if not green_text_has_explicit_time(green_text, item["time"]):
        raise ValueError(
            "绿色气泡必须直接写出无歧义的具体钟点，可使用 HH:MM 或自然中文时间"
        )

    task_text = re.sub(r"[^\w\u4e00-\u9fff]+", "", item.get("task", ""))
    task_bigrams = {
        task_text[index : index + 2]
        for index in range(max(0, len(task_text) - 1))
        if task_text[index : index + 2]
        not in {"安排", "事情", "任务", "处理", "确认"}
    }
    compact_green_text = re.sub(r"[^\w\u4e00-\u9fff]+", "", green_text)
    if not task_bigrams or not any(
        fragment in compact_green_text for fragment in task_bigrams
    ):
        raise ValueError(
            "绿色气泡必须直接说明 schedule.task 的具体动作，不能只回复‘好’或‘到时见’"
        )


def validate_group_training_schedule(value: dict[str, Any]) -> None:
    """Require one future group agreement stated in green bubbles."""
    participants = value.get("participants")
    messages = value.get("messages")
    schedule = value.get("schedule")
    if not isinstance(participants, list) or not 3 <= len(participants) <= 5:
        raise ValueError("群聊明确约定校验需要 3–5 名参与者")
    if not isinstance(messages, list) or not messages:
        raise ValueError("群聊明确约定校验需要有效消息")
    if not isinstance(schedule, list) or len(schedule) != 1:
        raise ValueError(
            "群聊训练样本必须有且仅有 1 项明确未来约定，schedule 不能是空数组"
        )

    green_name = re.split(r"[（(]", participants[0], maxsplit=1)[0].strip()
    item = schedule[0]
    if not isinstance(item, dict) or item.get("owner") != green_name:
        raise ValueError(
            "群聊 schedule.owner 必须是 participants[0]（右侧绿色气泡用户）"
        )

    try:
        scheduled_at = datetime.datetime.strptime(
            f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M"
        )
        latest_message_at = max(
            datetime.datetime.strptime(message["time"], "%Y-%m-%d %H:%M")
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("time"), str)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("无法核对群聊约定与消息时间") from error
    if scheduled_at <= latest_message_at:
        raise ValueError("群聊 schedule 必须是晚于聊天结束时间的未来约定")

    green_text = " ".join(
        message.get("text", "")
        for message in messages
        if isinstance(message, dict)
        and message.get("speaker") == green_name
        and isinstance(message.get("text"), str)
    )
    schedule_date = datetime.datetime.strptime(item["date"], "%Y-%m-%d")
    date_tokens = (
        item["date"],
        f"{schedule_date.year}年{schedule_date.month}月{schedule_date.day}日",
        f"{schedule_date.month}月{schedule_date.day}日",
        f"{schedule_date.month}月{schedule_date.day}号",
    )
    if not any(token in green_text for token in date_tokens):
        raise ValueError("群聊绿色气泡必须直接写出最终约定的 M月D日")
    if not green_text_has_explicit_time(green_text, item["time"]):
        raise ValueError(
            "群聊绿色气泡必须直接写出无歧义的具体钟点，可使用 HH:MM 或自然中文时间"
        )

    task_text = re.sub(r"[^\w\u4e00-\u9fff]+", "", item.get("task", ""))
    task_bigrams = {
        task_text[index : index + 2]
        for index in range(max(0, len(task_text) - 1))
        if task_text[index : index + 2]
        not in {"安排", "事情", "任务", "处理", "确认"}
    }
    compact_green_text = re.sub(r"[^\w\u4e00-\u9fff]+", "", green_text)
    if not task_bigrams or not any(
        fragment in compact_green_text for fragment in task_bigrams
    ):
        raise ValueError("群聊绿色气泡必须直接说明 schedule.task 的具体动作")


def validate_calendar_mentions(value: dict[str, Any]) -> None:
    """Ensure explicit calendar dates agree with Chinese weekday words."""
    weekday_names = "一二三四五六日"
    schedule = value.get("schedule")
    if not isinstance(schedule, list):
        return
    schedule_dates: list[datetime.datetime] = []
    for item in schedule:
        if not isinstance(item, dict) or not isinstance(item.get("date"), str):
            continue
        try:
            schedule_dates.append(
                datetime.datetime.strptime(item["date"], "%Y-%m-%d")
            )
        except ValueError:
            continue
    years = {item.year for item in schedule_dates}
    if len(years) != 1:
        return
    year = next(iter(years))

    messages = value.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("text"), str
        ):
            continue
        text = message["text"]
        for match in re.finditer(
            r"(\d{1,2})月(\d{1,2})日?(?:周|星期)([一二三四五六日天])", text
        ):
            month, day = int(match.group(1)), int(match.group(2))
            weekday = "日" if match.group(3) == "天" else match.group(3)
            try:
                expected = weekday_names[
                    datetime.datetime(year, month, day).weekday()
                ]
            except ValueError as error:
                raise ValueError(
                    f"消息中的 {month}月{day}日 不是有效日期"
                ) from error
            if weekday != expected:
                raise ValueError(
                    f"消息中的 {year}年{month}月{day}日应为周{expected}，不能写周{weekday}"
                )

    unique_schedule_days = {item.date() for item in schedule_dates}
    if len(unique_schedule_days) == 1 and messages:
        final_text = (
            messages[-1].get("text", "")
            if isinstance(messages[-1], dict)
            else ""
        )
        matches = re.findall(r"(?:周|星期)([一二三四五六日天])", final_text)
        if matches:
            weekday = "日" if matches[-1] == "天" else matches[-1]
            expected = weekday_names[schedule_dates[0].weekday()]
            if weekday != expected:
                raise ValueError(f"最终确认应为周{expected}，不能写周{weekday}")


def normalize_calendar_mentions(value: dict[str, Any]) -> dict[str, Any]:
    """Remove incorrect weekday words in place; keep dates and times."""
    schedule = value.get("schedule")
    messages = value.get("messages")
    if not isinstance(schedule, list) or not isinstance(messages, list):
        return value
    schedule_dates: list[datetime.datetime] = []
    for item in schedule:
        if not isinstance(item, dict) or not isinstance(item.get("date"), str):
            continue
        try:
            schedule_dates.append(
                datetime.datetime.strptime(item["date"], "%Y-%m-%d")
            )
        except ValueError:
            continue
    years = {item.year for item in schedule_dates}
    if len(years) != 1:
        return value
    year = next(iter(years))
    weekday_names = "一二三四五六日"

    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("text"), str
        ):
            continue

        def strip_wrong_explicit_weekday(match: re.Match[str]) -> str:
            month, day = int(match.group(1)), int(match.group(2))
            weekday = "日" if match.group(3) == "天" else match.group(3)
            try:
                expected = weekday_names[
                    datetime.datetime(year, month, day).weekday()
                ]
            except ValueError:
                return match.group(0)
            return (
                f"{month}月{day}日" if weekday != expected else match.group(0)
            )

        message["text"] = re.sub(
            r"(\d{1,2})月(\d{1,2})日?(?:周|星期)([一二三四五六日天])",
            strip_wrong_explicit_weekday,
            message["text"],
        )

    unique_schedule_days = {item.date() for item in schedule_dates}
    if (
        len(unique_schedule_days) == 1
        and messages
        and isinstance(messages[-1], dict)
    ):
        final_text = messages[-1].get("text")
        if isinstance(final_text, str):
            expected = weekday_names[schedule_dates[0].weekday()]

            def strip_wrong_final_weekday(match: re.Match[str]) -> str:
                weekday = "日" if match.group(1) == "天" else match.group(1)
                return match.group(0) if weekday == expected else ""

            messages[-1]["text"] = re.sub(
                r"(?:周|星期)([一二三四五六日天])",
                strip_wrong_final_weekday,
                final_text,
            )
    return value


def rebuild_group_participants(value: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the group roster in place from formal speaker names."""
    participants = value.get("participants")
    messages = value.get("messages")
    if not isinstance(participants, list) or not isinstance(messages, list):
        return value

    roles: dict[str, str] = {}
    for participant in participants:
        if not isinstance(participant, str):
            continue
        match = re.fullmatch(
            r"([^（(]+)[（(]([^）)]+)[）)]", participant.strip()
        )
        if match:
            roles[match.group(1).strip()] = match.group(2).strip()

    formal_roster = [name for name in roles if is_formal_chinese_name(name)]

    def match_reference(reference: str, used: list[str]) -> str | None:
        if reference in formal_roster:
            return reference
        stem = reference
        if stem.startswith(NAME_PREFIXES):
            stem = stem[1:]
        for suffix in NAME_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem in formal_roster:
            return stem
        surname_matches = [
            name for name in formal_roster if stem and name.startswith(stem[0])
        ]
        if len(surname_matches) == 1:
            return surname_matches[0]
        unused = [name for name in formal_roster if name not in used]
        return unused[0] if len(unused) == 1 else None

    speaker_names: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
            message.get("speaker"), str
        ):
            return value
        speaker = message["speaker"].strip()
        matched = match_reference(speaker, speaker_names)
        if matched:
            speaker = matched
            message["speaker"] = matched
        if not is_formal_chinese_name(speaker):
            return value
        if speaker not in speaker_names:
            speaker_names.append(speaker)

    if not 3 <= len(speaker_names) <= 5:
        return value
    value["participants"] = [
        f"{name}（{roles.get(name, '群成员')}）" for name in speaker_names
    ]

    schedule = value.get("schedule")
    if isinstance(schedule, list):
        valid_schedule: list[dict[str, Any]] = []
        for item in schedule:
            if not isinstance(item, dict) or not isinstance(
                item.get("owner"), str
            ):
                continue
            owner = match_reference(item["owner"].strip(), speaker_names)
            if owner in speaker_names:
                item["owner"] = owner
                valid_schedule.append(item)
        value["schedule"] = valid_schedule
    return value


def group_plan_contract(
    plan: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return a canonical roster and speaker sequence from a group beat plan."""
    if not isinstance(plan, dict):
        raise ValueError("群聊节拍规划必须是 JSON 对象")
    participants = plan.get("participants")
    beats = plan.get("beats")
    if not isinstance(beats, list) or not 10 <= len(beats) <= 12:
        raise ValueError("群聊节拍规划必须包含 10–12 个节拍")

    names: list[str] = []
    roles: dict[str, str] = {}
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            name = participant.get("name")
            if not is_formal_chinese_name(name) or name in names:
                continue
            role = participant.get("group_role")
            names.append(name)
            roles[name] = role.strip() if _nonempty_string(role) else "群成员"

    raw_speakers: list[str] = []
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict) or not _nonempty_string(
            beat.get("speaker")
        ):
            raise ValueError(
                f"群聊节拍 beats[{index}].speaker 必须是非空字符串"
            )
        raw_speakers.append(beat["speaker"].strip())

    def match_roster_name(reference: str) -> str | None:
        if reference in names:
            return reference
        stem = reference
        if stem.startswith(NAME_PREFIXES):
            stem = stem[1:]
        for suffix in NAME_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem in names:
            return stem
        matches = [name for name in names if stem and name.startswith(stem[0])]
        return matches[0] if len(matches) == 1 else None

    if 3 <= len(names) <= 5:
        matched_speakers = [
            match_roster_name(speaker) for speaker in raw_speakers
        ]
        if all(matched_speakers) and len(set(matched_speakers)) >= 3:
            return (
                names,
                [
                    speaker
                    for speaker in matched_speakers
                    if speaker is not None
                ],
                roles,
            )

    # Some model responses omit or over-expand participants while their beat
    # speakers are already a complete formal roster. Recover that roster locally.
    beat_names: list[str] = []
    if all(is_formal_chinese_name(speaker) for speaker in raw_speakers):
        for speaker in raw_speakers:
            if speaker not in beat_names:
                beat_names.append(speaker)
    if 3 <= len(beat_names) <= 5:
        recovered_roles = {
            name: roles.get(name, "群成员") for name in beat_names
        }
        return beat_names, raw_speakers, recovered_roles

    if not 3 <= len(names) <= 5:
        raise ValueError("群聊节拍规划必须包含 3–5 名参与者")
    raise ValueError(
        "群聊节拍 speaker 无法映射到参与者白名单，或实际发言少于 3 人"
    )


def private_brief_contract(
    brief: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Extract the two-person roster and identities from a private brief."""
    if not isinstance(brief, dict) or not isinstance(
        brief.get("participants"), list
    ):
        raise ValueError("私聊蓝图必须包含 participants 数组")
    participants = brief["participants"]
    if len(participants) != 2:
        raise ValueError("私聊蓝图 participants 必须恰好有 2 人")

    names: list[str] = []
    identities: dict[str, str] = {}
    for index, participant in enumerate(participants):
        if not isinstance(participant, dict):
            raise ValueError(f"私聊蓝图 participants[{index}] 必须是对象")
        name = participant.get("name")
        identity = participant.get("identity")
        if not is_formal_chinese_name(name):
            raise ValueError(
                f"私聊蓝图 participants[{index}].name 不是正式中文姓名"
            )
        if name in names:
            raise ValueError("私聊蓝图参与者姓名不能重复")
        if not _nonempty_string(identity):
            raise ValueError(
                f"私聊蓝图 participants[{index}].identity 不能为空"
            )
        names.append(name)
        identities[name] = identity.strip()
    return names, identities


def group_brief_contract(
    brief: dict[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Extract the 3–5 person roster and roles from a group brief."""
    if not isinstance(brief, dict) or not isinstance(
        brief.get("participants"), list
    ):
        raise ValueError("群聊蓝图必须包含 participants 数组")
    participants = brief["participants"]
    if not 3 <= len(participants) <= 5:
        raise ValueError("群聊蓝图 participants 必须有 3–5 人")

    names: list[str] = []
    roles: dict[str, str] = {}
    for index, participant in enumerate(participants):
        if not isinstance(participant, dict):
            raise ValueError(f"群聊蓝图 participants[{index}] 必须是对象")
        name = participant.get("name")
        role = participant.get("group_role")
        if not is_formal_chinese_name(name):
            raise ValueError(
                f"群聊蓝图 participants[{index}].name 不是正式中文姓名"
            )
        if name in names:
            raise ValueError("群聊蓝图参与者姓名不能重复")
        if not _nonempty_string(role):
            raise ValueError(
                f"群聊蓝图 participants[{index}].group_role 不能为空"
            )
        names.append(name)
        roles[name] = role.strip()
    return names, roles


def private_plan_contract(
    plan: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return the locked private roster and exact speaker sequence."""
    names, identities = private_brief_contract(plan)
    beats = plan.get("beats")
    if not isinstance(beats, list) or not 10 <= len(beats) <= 12:
        raise ValueError("私聊节拍规划必须包含 10–12 个节拍")
    speakers: list[str] = []
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict) or beat.get("speaker") not in names:
            raise ValueError(
                f"私聊节拍 beats[{index}].speaker 不在姓名白名单中"
            )
        speakers.append(beat["speaker"])
    if len(set(speakers)) != 2:
        raise ValueError("私聊节拍必须由两位参与者共同发言")
    return names, speakers, identities


def apply_private_plan_constraints(
    value: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Apply private identities in place and return the same conversation."""
    names, planned_speakers, identities = private_plan_contract(plan)
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) != len(planned_speakers):
        return value
    for message, speaker in zip(messages, planned_speakers, strict=True):
        if not isinstance(message, dict):
            return value
        message["speaker"] = speaker
    value["participants"] = [f"{name}（{identities[name]}）" for name in names]

    schedule = value.get("schedule")
    if isinstance(schedule, list):
        value["schedule"] = [
            item
            for item in schedule
            if isinstance(item, dict) and item.get("owner") in names
        ]
    return value


def apply_group_plan_constraints(
    value: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """Apply the group plan in place, dropping unresolvable schedule owners."""
    names, planned_speakers, roles = group_plan_contract(plan)
    messages = value.get("messages")
    if not isinstance(messages, list) or not 10 <= len(messages) <= 12:
        return value

    # The prompt requires one message per beat. Correcting identities locally
    # prevents the review model from inventing a nickname or a sixth member.
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return value
        planned_index = min(index, len(planned_speakers) - 1)
        message["speaker"] = planned_speakers[planned_index]

    value["participants"] = [f"{name}（{roles[name]}）" for name in names]

    schedule = value.get("schedule")
    if isinstance(schedule, list):
        normalized_schedule: list[dict[str, Any]] = []
        for item in schedule:
            if not isinstance(item, dict) or not isinstance(
                item.get("owner"), str
            ):
                continue
            owner = item["owner"].strip()
            if owner not in names:
                stem = owner
                if stem.startswith(NAME_PREFIXES):
                    stem = stem[1:]
                for suffix in NAME_SUFFIXES:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                matches = [
                    name
                    for name in names
                    if stem and (stem == name or name.startswith(stem[0]))
                ]
                if len(matches) != 1:
                    continue
                owner = matches[0]
            item["owner"] = owner
            normalized_schedule.append(item)
        value["schedule"] = normalized_schedule
    return value


DatasetGenerator = Callable[[ChatCompletionClient, str, int], dict[str, Any]]


def generate_dataset(
    chat_type: ChatType,
    count: int,
    output: pathlib.Path,
    *,
    generator: DatasetGenerator,
    direction: str | None = None,
    delay: float = 0.2,
    client: ChatCompletionClient | None = None,
    sequence_offset: int = 0,
) -> list[dict[str, Any]]:
    """Generate or resume a dataset, saving each accepted conversation.

    Args:
        chat_type: Conversation schema to generate and validate.
        count: Target total size, including rows already present in output.
        output: JSON array to resume from and replace after each accepted row.
        generator: Prompt-chain callback receiving a client, direction, and
            one-based sequence number.
        direction: Optional creative constraints for the prompt chain.
        delay: Seconds to wait between conversations and failed attempts.
        client: API client; defaults to a new DeepSeek client.
        sequence_offset: Offset added to sequence numbers passed to generator.

    Returns:
        Existing validated rows followed by newly generated conversations.

    Raises:
        ValueError: The target count or existing dataset is invalid.
        RuntimeError: A conversation exhausts its generation attempts. Earlier
            accepted rows remain saved for a later run.
    """
    if count < 1:
        raise ValueError("count 必须至少为 1")
    api = client or ChatCompletionClient()
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if output.exists():
        loaded = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"现有输出 {output} 顶层必须是 JSON 数组")
        for item in loaded:
            validated = validate_conversation(item, chat_type)
            validate_calendar_mentions(validated)
            validate_natural_dialogue(validated)
            if chat_type == "group":
                validate_closed_group_ending(validated)
            results.append(validated)
        if len(results) > count:
            raise ValueError(
                f"现有输出已有 {len(results)} 条，超过目标数量 {count}"
            )
        if results:
            print(f"断点续跑：已存在 {len(results)}/{count} 条合格数据")
    for index in range(len(results), count):
        creative_request = (
            direction.strip()
            if direction and direction.strip()
            else (
                "开放创作：自主选择自然、可信且适合微信聊天的人物关系与事件，"
                "不要从预设主题列表、固定情节、词库或句式中抽取和拼接。"
            )
        )
        recent_topics = [
            item["topic"]
            for item in results[-8:]
            if isinstance(item.get("topic"), str)
        ]
        if recent_topics:
            creative_request += (
                "\n本批次最近已经生成的主题为："
                f"{compact_json(recent_topics)}。这是硬性避重条件：先概括这些主题所属的关系、生活领域、"
                "核心动作和主要物件，再选择一个均不相同的新场景；同义改写、把蛋糕换成曲奇、把爬山"
                "换成露营之类只换表面元素不算新主题。不要在回复中输出分析过程。"
            )
        recent_domain_tags = sorted(
            {
                domain
                for item in results[-DOMAIN_RECENCY_WINDOW:]
                for domain in conversation_primary_domain_tags(item)
            }
        )
        if recent_domain_tags:
            creative_request += (
                f"\n本地分类器判定最近 {DOMAIN_RECENCY_WINDOW} 条已覆盖这些领域：{compact_json(recent_domain_tags)}。"
                "本条不得继续使用其中任一领域；这些标签只用于避重，不是让你从其他标签中机械选题。"
            )
        if chat_type == "group":
            recent_group_names = [
                item["group_name"]
                for item in results[-24:]
                if isinstance(item.get("group_name"), str)
            ]
            if recent_group_names:
                creative_request += (
                    "\n最近已经使用的群名为："
                    f"{compact_json(recent_group_names)}。新 group_name 不得与其中任何一个重复，"
                    "也不要只替换‘周末、搭子、小分队、行动组’等高频词重新拼一个近似名称。"
                    "如果这些群名明显属于相同场域或关系，本条必须换到不同的关系和生活领域，"
                    "而不只是换一个地点名。"
                )
        print(f"[{index + 1}/{count}] 开放创作并执行提示链")
        last_error: Exception | None = None
        failure_notes: list[str] = []
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                attempt_request = creative_request
                if attempt > 1 and direction is None:
                    available_hints = [
                        label
                        for label in RETRY_DOMAIN_HINTS
                        if label not in recent_domain_tags
                    ]
                    if available_hints:
                        hint = available_hints[
                            (index + attempt - 2) % len(available_hints)
                        ]
                        attempt_request += (
                            f"\n本次重试优先从“{hint}”这一宽泛生活领域发散原创场景；"
                            "它只是多样性方向，不得套用固定人物、固定剧情或固定句式。"
                        )
                if failure_notes:
                    attempt_request += (
                        "\n最近被本地校验拒绝的原因："
                        f"{compact_json(failure_notes[-3:])}。必须逐项避开，不能原样重试。"
                    )
                result = generator(
                    api, attempt_request, index + 1 + sequence_offset
                )
                topic = result.get("topic")
                if not isinstance(topic, str) or not topic.strip():
                    raise ValueError("生成结果缺少 topic")
                risk_text = compact_json(result)
                if re.search(
                    r"野生菌|毒蘑菇|误食|吞食|催吐|食物中毒|急救|病人|输液|查房|自行停药|民间偏方|荐股|高息借贷",
                    risk_text,
                ):
                    raise ValueError(f"主题不符合普通低风险场景要求：{topic}")
                topic_key = re.sub(r"[^\w\u4e00-\u9fff]+", "", topic).lower()
                for prior_topic in recent_topics:
                    prior_key = re.sub(
                        r"[^\w\u4e00-\u9fff]+", "", prior_topic
                    ).lower()
                    if topic_key == prior_key:
                        raise ValueError(f"主题与已有数据重复：{topic}")
                    if min(len(topic_key), len(prior_key)) >= 8:
                        similarity = difflib.SequenceMatcher(
                            None, topic_key, prior_key
                        ).ratio()
                        if similarity >= 0.5:
                            raise ValueError(
                                f"主题与已有数据过于相似（{similarity:.0%}）：{topic} / {prior_topic}"
                            )
                current_domains = conversation_primary_domain_tags(result)
                for prior in results[-DOMAIN_RECENCY_WINDOW:]:
                    prior_domains = conversation_primary_domain_tags(prior)
                    overlap = current_domains & prior_domains
                    if overlap:
                        raise ValueError(
                            f"生活领域与最近数据重复（{','.join(sorted(overlap))}）：{topic}"
                        )
                if chat_type == "group":
                    group_name = result.get("group_name")
                    used_group_names = {
                        item.get("group_name")
                        for item in results
                        if isinstance(item.get("group_name"), str)
                    }
                    if (
                        not isinstance(group_name, str)
                        or not group_name.strip()
                    ):
                        raise ValueError("群聊生成结果缺少 group_name")
                    if group_name.strip() in used_group_names:
                        raise ValueError(
                            f"群名与已有数据重复：{group_name.strip()}"
                        )
                messages = result.get("messages")
                if (
                    isinstance(messages, list)
                    and messages
                    and isinstance(messages[-1], dict)
                ):
                    final_text = messages[-1].get("text")
                    if isinstance(final_text, str):
                        ending_key = re.sub(
                            r"[\s，。！？!?～~]+", "", final_text
                        )
                        used_endings = {
                            re.sub(
                                r"[\s，。！？!?～~]+",
                                "",
                                item["messages"][-1]["text"],
                            )
                            for item in results
                            if isinstance(item.get("messages"), list)
                            and item["messages"]
                            and isinstance(item["messages"][-1], dict)
                            and isinstance(
                                item["messages"][-1].get("text"), str
                            )
                        }
                        if (
                            ending_key
                            and ending_key not in GENERIC_ENDING_KEYS
                            and ending_key in used_endings
                        ):
                            raise ValueError(
                                f"结尾与已有数据重复：{final_text}"
                            )
                results.append(result)
                # Save each accepted row so API failures can be resumed.
                temp_output = output.with_suffix(output.suffix + ".tmp")
                temp_output.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp_output, output)
                break
            except (ValueError, RuntimeError) as error:
                last_error = error
                print(f"  第 {attempt} 次失败：{error}", file=sys.stderr)
                if attempt < MAX_GENERATION_ATTEMPTS:
                    note = str(error)[:240]
                    if note not in failure_notes:
                        failure_notes.append(note)
                    time.sleep(max(delay, 0))
        else:
            raise RuntimeError(
                f"当前会话连续 {MAX_GENERATION_ATTEMPTS} 次生成失败：{last_error}"
            )
        if delay > 0 and index + 1 < count:
            time.sleep(delay)
    return results


def dataset_cli(
    chat_type: ChatType,
    generator: DatasetGenerator,
    argv: list[str] | None = None,
) -> int:
    """Run generation or model listing and return a command-line exit status."""
    label = "私聊" if chat_type == "private" else "群聊"
    parser = argparse.ArgumentParser(
        description=f"使用 DeepSeek、CLIProxyAPI 或自定义接口提示链生成{label} JSON 数据集"
    )
    add_model_arguments(parser)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="仅列出本机 CLIProxyAPI 当前可用模型后退出",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=5, help="生成数量（默认：5）"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(f"data/{chat_type}_dataset.json"),
    )
    parser.add_argument(
        "--direction",
        help="可选的自由创作方向；不填写时由模型自主选择关系、场景和事件",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="每条数据之间的等待秒数"
    )
    parser.add_argument(
        "--sequence-offset",
        type=int,
        default=0,
        help="临时分片的序号偏移；普通生成保持默认 0",
    )
    args = parser.parse_args(argv)
    try:
        if args.list_models:
            if args.provider == "custom" or args.base_url is not None:
                raise ValueError("--list-models 仅用于本机 CLIProxyAPI")
            for model_id in list_cliproxy_models():
                print(model_id)
            return 0
        client = build_generation_client(
            args.provider, model=args.model, base_url=args.base_url
        )
        rows = generate_dataset(
            chat_type,
            args.count,
            args.output,
            generator=generator,
            direction=args.direction,
            delay=args.delay,
            client=client,
            sequence_offset=args.sequence_offset,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    print(f"完成：{len(rows)} 条{label}数据 -> {args.output}")
    return 0
