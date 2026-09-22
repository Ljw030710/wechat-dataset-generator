# 使用指南

[返回项目首页](../README.md) · [技术设计](architecture.md)

这套工具完成两件事：

1. 使用私聊或群聊专用的 Prompt Chaining，通过 DeepSeek、CLIProxyAPI 或自定义兼容接口生成固定 JSON 格式的聊天数据集。
2. 把 JSON 分别渲染成微信风格的私聊或群聊 PNG 截图。

生成器不从固定主题、人物、地点、物品或句式池中随机搭配。模型先开放构思场景，再使用“人物关系 + 聊天目的 + 具体事件 + 情绪变化 + 细节信息”等结构完成规划、成稿与审校。每段对话限制为 10–12 条短消息；截图采用 iPhone X 的 `1125:2436` 屏幕比例，默认固定为 `900×1949`，不会随对话变成长图或短图。

## 项目目录

```text
Createdate/
├── README.md                 # 项目介绍和快速开始
├── LICENSE                   # MIT 许可证
├── pyproject.toml            # Python 依赖及工具配置
├── uv.lock                   # 依赖版本锁定
├── scripts/                  # 生成、渲染和导出脚本
├── tests/                    # 自动化测试
├── examples/                 # 两段虚构演示 JSON
├── docs/                     # 使用文档和预览图
└── assets/                   # 头像和界面素材及来源说明
```

本项目发布生成工具，不分发作者的工作数据或训练数据集。
`data/`、`output/`、`llamafactory_*/`、`work/` 和 `archive/` 是本地工作目录，
已在 `.gitignore` 中排除；运行相应生成或导出命令时会按需创建输出目录。
首次体验请使用 [README 中的快速开始](../README.md#快速开始)，无需下载完整数据集。

所有命令均在项目根目录执行，脚本入口统一使用 `scripts/` 前缀。
`data/`、`output/` 等相对路径仍以运行命令时的目录为准。

`scripts/` 中的文件按用途分为：

| 用途 | 文件 |
| --- | --- |
| 生成私聊、群聊 | `generate_private_dataset.py`、`generate_group_dataset.py` |
| 渲染私聊、群聊 | `render_private_chat.py`、`render_group_chat.py` |
| 导出训练数据 | `export_llamafactory_dataset.py`、`export_group_llamafactory_dataset.py` |
| 合并与整理数据 | `merge_llamafactory_datasets.py`、`organize_all_datasets.py` |
| 追加会话数据 | `append_conversations.py` |
| 公共能力 | `dataset_generation.py`、`annotation_cache.py`、`chat_render_cli.py`、`wechat_screenshot.py` |

`.venv/` 是本地虚拟环境，`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`
是工具自动生成的缓存，已在 `.gitignore` 中忽略。

## 安装

需要 Python 3.11 或更高版本。项目使用 `uv`：

```bash
uv sync
```

## API Key 配置

推荐配置一次本地 `.env`，以后直接运行。先在项目根目录执行：

```bash
cp -n .env.example .env
```

用编辑器打开 `.env`，填写需要使用的一组配置。默认 DeepSeek 只需填写密钥：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-chat
```

保存后直接运行，下次打开终端也不用重新配置：

```bash
uv run python scripts/generate_private_dataset.py -n 10
```

私聊、群聊生成器及导出器的 `--label-mode teacher` 共用此配置。`LLM_PROVIDER` 可选 `deepseek`、`cliproxy`、`custom`，未配置时仍默认 DeepSeek。
配置优先级为：**显式命令行参数 > 终端环境变量 > 项目根目录 `.env` > 内置默认值**。密钥没有命令行参数，仍可沿用之前的 `export` 方式。
程序按脚本位置定位项目根目录，不读取其他工作目录的 `.env`；解析配置不会修改进程环境，也不会展开值中的 `${...}`。
若终端中曾设置同名变量，修改 `.env` 后需先 `unset` 对应变量，或打开没有预设变量的新终端。

密钥未设置或为空时会在发送请求前报错。本地截图、`schedule` 标注和测试不需要密钥。
`.gitignore` 已排除 `.env`，只发布不含真实密钥的 `.env.example` 模板。

如果密钥曾经暴露，应在提供商控制台撤销并重新生成。

## 可选：配置 CLIProxyAPI

默认的 CLIProxyAPI 地址和模型写在 [dataset_generation.py](../scripts/dataset_generation.py)：

```python
CLIPROXY_API_URL = "http://127.0.0.1:8317/v1/chat/completions"
CLIPROXY_MODEL = "gemini-3.7-flash-high"
```

API Key 优先读取 `CLIPROXY_API_KEY` 环境变量；未设置时读取 Homebrew 配置
`/opt/homebrew/etc/cliproxyapi.conf` 中的第一项 `api-keys`。运行生成命令时明确传入
`--provider cliproxy`，就只会访问本机 CLIProxyAPI，不会请求 DeepSeek。`sol` 仍可作为
旧命令的兼容别名。

查看代理当前实际可用的模型（代理配置变更后也适用）：

```bash
uv run python scripts/generate_private_dataset.py --provider cliproxy --list-models
```

默认使用 `gemini-3.7-flash-high`。如需切换，在命令中传入模型 ID：

```bash
uv run python scripts/generate_private_dataset.py --provider cliproxy --model gemini-3.7-flash-high -n 10 -o data/private_dataset.json
```

也可在 `.env` 中设置 `LLM_PROVIDER=cliproxy`、`CLIPROXY_API_KEY` 和 `CLIPROXY_MODEL`，后续省略命令中的服务商和模型参数。

## 自定义模型服务

保留默认 DeepSeek 和 CLIProxyAPI 的同时，`--provider custom` 可以连接其他兼容服务。
生成器和 `teacher` 导出共用以下配置：

| 模式 | 密钥 | 接口地址 | 模型 |
| --- | --- | --- | --- |
| `deepseek`（默认） | `DEEPSEEK_API_KEY` | 内置 DeepSeek 地址 | `DEEPSEEK_MODEL` 或 `--model`，默认 `deepseek-chat` |
| `cliproxy` | `CLIPROXY_API_KEY` 或代理配置文件 | 本机 CLIProxyAPI | `CLIPROXY_MODEL` 或 `--model` |
| `custom` | `LLM_API_KEY` | `LLM_BASE_URL` 或 `--base-url` | `LLM_MODEL` 或 `--model` |

在项目根目录的 `.env` 中填写以下配置，将密钥、地址和模型替换为实际值：

```dotenv
LLM_PROVIDER=custom
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-id
```

随后生成私聊或群聊：

```bash
uv run python scripts/generate_private_dataset.py \
  -n 1 -o data/custom_private.json

uv run python scripts/generate_group_dataset.py \
  -n 1 -o data/custom_group.json
```

`--base-url` 和 `--model` 优先于对应环境变量。基础地址需要包含服务商要求的路径，例如 `/v1` 或 `/compatible-mode/v1`；程序追加 `/chat/completions`，也接受已包含该后缀的完整地址。
密钥只从 `LLM_API_KEY` 读取，不要求以 `sk-` 开头，不会回退到 DeepSeek 的密钥。自定义地址仅在选择 `custom` 时使用。

兼容接口需要接受 Bearer 鉴权、`messages`、`temperature`、`stream: false` 和 `response_format: {"type": "json_object"}`，并在 `choices[0].message.content` 返回 JSON 文本。
项目当前没有为不同服务商自动改写协议或移除不支持的参数；使用其他原生协议的接口需要额外适配。
`--list-models` 仍仅查询本机 CLIProxyAPI，自定义模型 ID 请从服务商获取。

`teacher` 标注同样支持这些模型参数。例如，已有私聊 JSON 和截图时：

```bash
uv run python scripts/export_llamafactory_dataset.py \
  --source data/custom_private.json \
  --images output/custom_private \
  --output output/custom_teacher \
  --label-mode teacher --provider custom
```

上例的截图需先用 `render_private_chat.py` 渲染到 `output/custom_private`。群聊使用对应的群聊导出脚本。
`schedule` 模式始终在本地运行，不创建模型客户端，也不要求配置密钥。
`schedule` 模式每次都根据当前源数据重新计算标签，可直接向同一输出目录重新导出。
`teacher` 模式使用 `teacher_cache.json` 保存每条标签及其请求指纹；只有源内容、服务商、接口地址、模型、标注提示和采样温度均未变化时才复用。修改其中任一项后，受影响的样本会重新调用模型标注，其他样本继续复用。
缓存不会记录 API Key，单独更换密钥不会触发重新标注。标注逻辑版本变化时也会使缓存失效。
旧版本的 `annotations_teacher.json` 没有请求指纹，下次 `teacher` 导出会重新标注并建立缓存，可能产生 API 费用。导出的 `annotations_*.json` 仍保持原格式，并只包含本次源数据中的会话。
修改对话正文后，需先重新渲染截图再导出；标签缓存不会检查图片与源对话是否一致。

## 第二步：生成固定格式数据集

私聊和群聊只使用两个独立生成入口。`generate_private_dataset.py` 包含私聊的五要素公式和完整提示链；`generate_group_dataset.py` 包含群聊的场景公式和完整提示链。`dataset_generation.py` 提供模型接口、固定 JSON 校验、重试和批量写入等公共能力，不负责提供题材或模板。

生成 10 条私聊：

```bash
uv run python scripts/generate_private_dataset.py --provider cliproxy -n 10 -o data/private_dataset.json
```

生成 10 条群聊：

```bash
uv run python scripts/generate_group_dataset.py --provider cliproxy -n 10 -o data/group_dataset.json
```

默认不限定题材。如果希望给模型一个开放方向，可以使用可选参数，例如：

```bash
uv run python scripts/generate_private_dataset.py --provider cliproxy -n 10 -o data/private_dataset.json \
  --direction "围绕不同年龄和关系中的真实沟通，不限定具体行业或事件"
```

`--direction` 只提供创作意图，不会变成固定词库。批量生成时，最近生成的主题会传给下一条，减少关系和事件重复。私聊数据会依次经过：

1. 五要素场景蓝图：自主构思人物关系、聊天目的、具体事件和一个未来仍需执行的明确约定，此时锁定绿色气泡负责人、任务、日期、时间和可选提醒，但不写台词。
2. 人物与节拍规划：继承蓝图，规划人物语言习惯、10–12 个消息节拍和小波折，并确保右侧绿色发送者亲自说出最终任务、日期和时间，此时仍不写完整台词。
3. 对话成稿：同时依据蓝图与节拍生成自然、简短的聊天 JSON；`schedule` 必须与蓝图中的最终约定完全一致。
4. 对照质检：再次读取原始蓝图、节拍和草稿，逐项检查内容覆盖、真实感与固定格式，并消除“时间不变”“还是那个安排”等无法直接抽取的模糊表达。

群聊使用独立公式“群聊关系 + 眼前事项 + 成员状态 + 明确约定 + 自然落点”，四段链依次负责场景蓝图、人物互动节拍、群聊成稿和对照质检。它会额外检查至少三人发言、成员位置与语气不同、避免机械轮流、信息逐步出现，并锁定绿色方明确参与的一项未来约定。

质检后还有本地硬校验。新生成的私聊和群聊都必须有且仅有一个未来约定，`owner` 必须是 `participants[0]`；该发送者的绿色气泡必须直接包含最终动作、`M月D日`/`M月D号` 和无歧义的具体钟点。消息可使用“上午九点半”“晚上七点四十”这类自然中文时间，`schedule.time` 仍规范化为 `HH:MM`。纯聊家常、空 `schedule`、只由白色气泡提供信息或约定早于聊天结束时间的样本会被拒绝并重试。

## 固定 JSON 格式

私聊顶层只能有以下 5 个字段；群聊会在末尾额外包含 `group_name`，作为截图标题使用。
下面仅展示字段结构，消息已省略，不能作为完整训练样本；可直接运行的完整数据见
[私聊示例](../examples/private_demo.json)和[群聊示例](../examples/group_demo.json)：

```json
{
  "conversation_id": "conv_private_20260712_101500_0001",
  "messages": [
    {
      "speaker": "小林",
      "text": "明天下午的电影先取消吧，我临时有事。",
      "time": "2026-07-12 10:15"
    }
  ],
  "participants": [
    "小林（临时有事）",
    "阿哲（负责订票）"
  ],
  "schedule": [
    {
      "date": "2026-07-18",
      "owner": "小林",
      "task": "重新购买电影票",
      "time": "17:00"
    }
  ],
  "topic": "朋友取消电影并改期"
}
```

固定规则：

- `conversation_id`、`messages`、`participants`、`schedule`、`topic` 不得缺少或增加。
- 群聊必须额外包含 `group_name`：2–12 个字符的正常群名，例如“周末搭子群”；它不能是事件摘要。`topic` 仍用于概括本次聊天内容。
- `messages` 固定 10–12 条；每条最多 28 个汉字，建议保持在 8–22 字。
- 消息字段固定为 `speaker`、`text`、`time`。
- 时间固定为 `YYYY-MM-DD HH:MM`，并按时间递增。
- 私聊恰好 2 位参与者；群聊为 3–5 位参与者。
- `speaker` 使用参与者括号前的姓名。
- `schedule` 只记录聊天结束后仍有效的最终安排。新生成的私聊和群聊都固定为 1 项明确约定。
- 日程字段固定为 `date`、`owner`、`task`、`time`。
- 不添加 `chat_type`。程序通过参与者人数判断私聊或群聊，因此仍兼容给定的示例结构。

数据集输出为 JSON 数组；截图脚本也兼容单个会话对象、JSONL 和包含 JSON 文件的目录。

## 第三步：生成截图

生成数据后，私聊数据使用私聊渲染脚本：

```bash
uv run python scripts/render_private_chat.py data/private_dataset.json -o output/private
```

群聊数据使用群聊渲染脚本：

```bash
uv run python scripts/render_group_chat.py data/group_dataset.json -o output/group
```

默认将 `participants[0]` 作为“我”，显示在右侧绿色气泡。可切换身份：

```bash
uv run python scripts/render_private_chat.py data/private_dataset.json \
  --self "阿哲" \
  -o output/private
```

群聊截图会显示其他成员的姓名；私聊截图不显示多余姓名标签。两个入口都会检查参与人数、消息数量和单条长度。遇到 10–12 条消息时会自动使用紧凑排版，以保持固定手机截图尺寸。

仍保留 `wechat_screenshot.py` 作为自动判断类型的兼容入口。它同样固定为 `1125:2436`，内容放不下时会要求缩短对话，不会扩展成长截图；新数据集仍建议使用上面的两个严格入口。

当前唯一保留的样式默认使用项目内的非人物动画头像素材板（只含动物、风景、植物和日常物品）。底部输入栏使用固定的 `assets/ui/wechat-footer.png` 整图素材，渲染时直接粘贴，不再用代码绘制按钮图标。

## 测试

```bash
uv run pytest -q
```

不需要模型 API Key 即可运行本地测试和截图生成。生成器和导出器的 `teacher` 模式会调用命令指定的模型提供商，默认使用 DeepSeek。
使用 `--provider cliproxy` 时，生成器连接本机代理，代理仍可能将请求转发到上游服务；这不代表模型在本机离线运行。

## 代码风格

Python 代码参照 [Google Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents.html)：
使用 4 空格缩进、80 字符目标行宽，按标准库、第三方库和项目模块分组导入。
通过模块名访问函数和类，类型注解所需的名称可以直接导入。

公共接口和复杂逻辑使用 Google 风格文档字符串；参数含义、返回值或异常不直观时，
补充 `Args`、`Returns`、`Raises`，并说明原地修改和文件写入等副作用。
行内注释解释约束或处理原因，不重复描述代码已经表达的操作。
模型提示词属于运行数据，保留原文及空白；不为满足行宽而改变提示词内容。

Ruff 配置和版本已纳入 `pyproject.toml` 与 `uv.lock`，覆盖格式、导入排序、
常见错误和文档字符串规范；业务语义和注释是否有用仍需人工检查。

```bash
uv run ruff format .
uv run ruff check .
```

只检查、不修改文件时运行 `uv run ruff format --check .`。

## 导出 LLaMA-Factory 图片文本对

截图工作流保持不变。私聊数据和截图生成完成后，运行独立导出器：

```bash
uv run python scripts/export_llamafactory_dataset.py \
  --source data/private_dataset.json \
  --images output/private \
  --output llamafactory_data \
  --label-mode schedule
```

导出器把 `participants[0]` 视为右侧绿色气泡发送者，使用源消息标注绿色气泡中最终有效的日程，
白色气泡只用于解析上下文。新版 Prompt Chain 会让动作、日期和自然钟点直接出现在绿色气泡中，
并让 `schedule.owner` 固定指向该发送者，从而减少截图内容与监督答案之间的歧义。监督输出固定为：

```json
{
  "items": [
    {
      "title": "在静安书店见面",
      "date": "2026-07-18",
      "time": "15:00",
      "note": "记得带上方案"
    }
  ]
}
```

`schedule` 模式完全在本地运行：只保留 `schedule.owner == participants[0]` 的事项，`note` 为空。
如需让标注模型阅读绿色气泡文本并补充 note，可使用
`--label-mode teacher`。`llamafactory_data/` 会包含复制后的图片、90/10 划分的训练/验证 JSON、
可断点续跑的标注文件，以及可直接放在 LLaMA-Factory `dataset_dir` 中使用的 `dataset_info.json`。
