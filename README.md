<h1 align="center">
  <img src="assets/logo.png" width="280" alt="Createdate">
</h1>

<p align="center">
  <strong>Createdate · 微信风格多模态合成数据生成器</strong>
</p>

<p align="center">
  一条命令生成中文私聊或群聊、微信风格截图与配套日程标签，导出 LLaMA-Factory 多模态数据格式。
</p>

<p align="center">
  无需 API Key 即可体验示例 · 支持 DeepSeek 与兼容接口 · MIT 开源
</p>

<p align="center">
  <strong>简体中文</strong> · <a href="README.en.md">English</a>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-3DA639" alt="Code License: MIT"></a>
</p>

<p align="center">
  <a href="#效果展示">效果展示</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#生成自己的数据">生成数据</a> ·
  <a href="docs/architecture.md">技术设计</a> ·
  <a href="docs/usage.md">使用指南</a> ·
  <a href="#许可">MIT 许可</a>
</p>

面向需要构建**中文聊天截图日程抽取数据**的开发者与研究者。
仓库提供从对话生成到图文样本导出的完整流程，可用于准备多模态微调数据。
仓库包含完整工具和两段虚构演示，不分发作者的工作数据或训练数据集。

## 效果展示

<div align="center">
  <table align="center">
    <tr>
      <th align="center">私聊 · 约定取画册</th>
      <th align="center">群聊 · 准备读书会海报</th>
    </tr>
    <tr>
      <td align="center"><img src="docs/images/private_demo.png" width="280" alt="AI 辅助编写的虚构私聊：约定去书店取画册"></td>
      <td align="center"><img src="docs/images/group_demo.png" width="280" alt="AI 辅助编写的虚构群聊：约定去书店贴海报"></td>
    </tr>
  </table>
  <p>以上截图由本项目渲染器生成，内容为 <strong>AI 辅助编写的虚构对话</strong>。</p>
</div>

左侧私聊截图对应的日程标签如下，导出时作为模型的目标回答：

```json
{
  "items": [
    {
      "title": "去书店取画册",
      "date": "2026-09-21",
      "time": "15:00",
      "note": ""
    }
  ]
}
```

## 快速开始

需要 **Python 3.11+**、[uv](https://docs.astral.sh/uv/getting-started/installation/)
和本地中文字体。首次使用可直接执行以下命令；已有本地项目时，从 `uv sync --locked` 开始：

```bash
# 克隆项目并进入目录
git clone https://github.com/Ljw030710/wechat-dataset-generator.git
cd wechat-dataset-generator

# 安装锁定版本的依赖
uv sync --locked

# 一条命令完成示例读取、截图渲染与训练数据导出（无需 API Key）
uv run python scripts/run_pipeline.py private \
  --source examples/private_demo.json -o output/demo
```

完成后可查看：

| 路径 | 内容 |
| --- | --- |
| `output/demo/images/private_demo.png` | 私聊截图 |
| `output/demo/llamafactory/images/` | 导出数据包中的配套图片 |
| `output/demo/llamafactory/*_train.json`、`*_eval.json` | 训练与验证样本 |
| `output/demo/llamafactory/dataset_info.json` | LLaMA-Factory 字段映射与数据集注册信息 |

**不需要 API Key 或模型服务。** 首次安装依赖需要联网。
演示只有一条样本，按当前划分逻辑进入验证集，训练集为空；它用于检查流程，正式训练需要自行生成足够的数据。

<details>
<summary>体验群聊完整流程</summary>

```bash
uv run python scripts/run_pipeline.py group \
  --source examples/group_demo.json -o output/demo/group
```

图片保存为 `output/demo/group/images/group_demo.png`，配套数据保存在
`output/demo/group/llamafactory/`。同样无需 API Key。

</details>

<details>
<summary>找不到中文字体时如何处理</summary>

通过 `--font` 指定已安装的中文 `.ttf` 或 `.ttc` 字体；将下方示例路径替换为实际路径：

```bash
uv run python scripts/run_pipeline.py private \
  --source examples/private_demo.json --font /path/to/chinese-font.ttf -o output/demo
```

字体不随项目分发，不同系统和字体的文字排版可能略有差异。

</details>

## 为什么使用 Createdate

- **图文配套**：从同一份会话生成私聊或群聊截图与日程标签，默认输出 `900 × 1949` 单屏 PNG，配套 ShareGPT 样本和 `dataset_info.json`。
- **完整流程**：一条命令完成生成、渲染与导出，也可分步运行；逐条保存会话，支持从已有数量继续生成。
- **质量检查**：检查参与者、日期和时间，再隔离创作背景核验标签的聊天原话依据；失败后反馈修复，结果仍需人工抽检。

## 生成自己的数据

首次使用时，将 [.env.example](.env.example) 复制为项目根目录的 `.env`，填写所选服务的密钥和模型配置：

```bash
cp -n .env.example .env
```

程序自动读取 `.env`。默认使用 DeepSeek；修改 `LLM_PROVIDER` 可选择 `cliproxy` 或 `custom`，
后续运行无需重复配置。填写方式见[密钥配置](docs/usage.md#api-key-配置)。

```bash
# 一次完成生成 → 渲染 → 导出
uv run python scripts/run_pipeline.py private \
  -n 1 --direction "朋友之间的日常协作" -o output/generated/private
```

结果保存在指定目录中的 `conversations.json`、`images/` 和 `llamafactory/`。
将 `private` 换成 `group` 可生成群聊；将 `-n 1` 改为目标总条数即可批量生成。
同一目录重跑会复用已保存的会话，重新渲染和导出。各步骤也可以独立运行，
参数与断点续跑说明见[完整流程入口](docs/usage.md#一条命令完成全流程)。

| 操作 | 是否调用模型 |
| --- | --- |
| 渲染已有 JSON、`schedule` 模式导出 | 否 |
| 生成新对话 | 是，使用所选提供商 |
| `teacher` 模式导出 | 是，将源对话发送给所选模型服务生成标注，默认 DeepSeek |

生成一段对话包含多次请求，校验失败还会重试，可能产生 API 费用。
也可使用 `--provider cliproxy` 连接自行配置的 CLIProxyAPI；本机代理仍可能转发请求到外部模型服务。
使用其他服务商时，在 `.env` 中设置 `LLM_PROVIDER=custom`，并填写 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。
接口需兼容 Chat Completions 和 JSON 对象输出，配置示例见[自定义模型服务](docs/usage.md#自定义模型服务)。
模型选择和命令参数见[使用指南](docs/usage.md)，提示链与质量控制原理见[技术设计](docs/architecture.md)。

## 技术设计与文档

生成采用 **Prompt Chaining**：场景蓝图 → 消息节拍 → 对话成稿 → 对照审校，
再进行标签证据核验，并结合近期话题与结尾去重减少重复。详细机制与适用边界见[技术设计](docs/architecture.md)。

| 入口 | 内容 |
| --- | --- |
| [使用指南](docs/usage.md) | 模型配置、完整 JSON 格式与命令 |
| [技术设计](docs/architecture.md) | Prompt Chaining、约束校验、反馈修复与图文数据配对 |
| [脚本](scripts/) | 对话生成、截图渲染、标注与数据导出 |
| [示例](examples/) | 可直接运行的私聊与群聊 JSON |
| [测试](tests/) | 本地自动化测试 |

## 数据与使用范围

- **抽取目标**：默认以 `participants[0]` 为右侧绿色气泡发送者，新生成数据要求此人明确参与一项未来约定。“未来”相对于对话时间。
- **任务范围**：当前生成流程集中于单项明确日程；无日程、多日程、模糊时间等场景需要额外构建。
- **格式与效果**：导出使用 LLaMA-Factory 的 ShareGPT 多模态格式；格式兼容不代表已经验证训练效果，项目尚无真实场景效果基准。
- **本地数据**：`data/`、`output/`、`archive/`、`llamafactory_*/` 和 `work/` 已在 `.gitignore` 中排除，不随源码发布。
- **合成说明**：程序不采集微信账户或聊天记录。截图目前不会自动添加合成水印，对外展示时请保留显著的虚构说明。

合成内容和标签仍需人工审核。用于训练或评估时，应补充负样本、多样化场景和独立测试集。
模型训练需要在 LLaMA-Factory 中另行配置，本项目负责数据准备。

## 贡献

欢迎提交 Issue 或 Pull Request，改进日程边界案例、排版效果、数据质量检查，
或补充可复现的训练与评估示例。反馈问题时请附上系统、Python 版本、运行命令和最小虚构 JSON。

本地测试无需模型 API Key：

```bash
uv run pytest -q
```

## 许可

代码、文档及 `examples/` 中的虚构演示采用 [MIT 许可证](LICENSE)，使用与分发时请保留版权及许可声明。

独立开发项目，与腾讯或微信无隶属、授权或背书关系。
第三方依赖与模型服务适用各自的许可或服务条款。
