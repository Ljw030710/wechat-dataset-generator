#!/usr/bin/env python3
"""Generate WeChat group datasets using the dedicated group-design chain."""

import datetime

import dataset_generation

GROUP_SCENE_FORMULA = "群聊关系 + 眼前事项 + 成员状态 + 明确约定 + 自然落点"
CONFIRMED_SCHEDULE_KEYS = ("owner", "task", "date", "time", "note")

GROUP_ROLE_GUIDANCE = """group_role 描述成员在这段聊天里的自然位置，不是工作职责，也不是任务分工。
	可按情节使用“先看到消息的人、知道内情的人、顺手补充的人、接梗的人、晚看到的人、平时潜水的人”
	等生活化描述。不要默认设置组织者、执行者和主持人；不要让每个人汇报自己负责什么。角色只是倾向，
	有人可以只回一句，也可以两个人聊了大半段，其他人后来才插进来。"""


def group_schedule_brief_contract(
    brief: dict, participant_names: list[str]
) -> dict:
    """Lock one extraction-ready green-user agreement before drafting."""
    item = brief.get("confirmed_schedule")
    if (
        not isinstance(item, dict)
        or tuple(item.keys()) != CONFIRMED_SCHEDULE_KEYS
    ):
        raise ValueError(
            f"群聊蓝图 confirmed_schedule 字段及顺序必须为 {CONFIRMED_SCHEDULE_KEYS}"
        )
    for key in ("owner", "task", "date", "time"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"群聊蓝图 confirmed_schedule.{key} 不能为空")
    if not isinstance(item.get("note"), str):
        raise ValueError("群聊蓝图 confirmed_schedule.note 必须是字符串")
    if item["owner"] != participant_names[0]:
        raise ValueError(
            "群聊蓝图的明确约定必须由 participants[0]（绿色气泡）负责或参与"
        )
    try:
        datetime.datetime.strptime(item["date"], "%Y-%m-%d")
        datetime.datetime.strptime(item["time"], "%H:%M")
    except ValueError as error:
        raise ValueError(
            "群聊蓝图的明确约定必须使用 YYYY-MM-DD 和 HH:MM"
        ) from error
    return item


def generate_group_one(
    client: dataset_generation.ChatCompletionClient,
    direction: str,
    sequence: int,
) -> dict:
    """Four-link group chain derived from the supplied group-design document."""
    message_count = 10 + (sequence - 1) % 3
    participant_count = 3 + (sequence - 1) % 3
    generation_date = datetime.datetime.now().date().isoformat()
    brief = client.complete(
        "你是微信群聊场景策划。你只设计群体聊天蓝图，不写对话，只输出合法 JSON。",
        f"""本次开放创作要求：{direction}

			请使用公式“{GROUP_SCENE_FORMULA}”原创一个适合单张微信截图、可用于日程抽取训练的群聊场景。
		先自主决定群聊关系、场域和正在聊的一件小事，不要从预设群聊类型中选择，也不要对固定人名、
	地点、物品或句式做随机替换。direction 如果提供了方向，只作为创作意图，不是固定剧情。
	direction 中的近期主题和群名是硬性反重复条件，不能用同义词或只换活动名称来规避。
	题材不同不等于事情越冷门越好。优先选择家人、同事、同学、邻居或固定爱好群里高频出现的
	眼前小事；不要为了显得新颖而设计需要成员解释专业流程、术语或完整背景的任务。熟人群默认
	共享很多上下文，不要让成员互相复述大家早已知道的来龙去脉。
		不要把群聊都写成周末聚会、选餐厅、采购分工或临时改期；真实群里也会有正在发生的小事、
		接前文的反应、临时求助和不需要建立完整项目计划的交流。

{GROUP_ROLE_GUIDANCE}

要求：
{dataset_generation.SCHEDULE_GROUNDING_RULES}
1. 本条必须使用恰好 {participant_count} 名成员，participants 数组长度必须等于 {participant_count}；
   不要求所有人同样活跃，但至少 3 人实际发言。
2. 在 participants 中一次确定全部成员；每项字段固定为 name, group_role, speaking_style,
   activity_level。name 只能是 2–3 个汉字的正式中文姓名（完整姓名）；禁止“小王”“阿杰”“老李”等昵称，
   禁止称谓、英文名、拼音、数字或“成员A”等代号。后续所有阶段不得改名或新增成员。
	3. common_task 只是“这几条消息眼下在聊什么”，可以是求助、见闻、反馈或一件小事，不等于集体任务。
	   不要把群聊写成项目管理会议。temporary_conflict 只在生活中自然发生时使用；
   没有明显冲突就写“无明显冲突”，禁止为了完整结构强造记错日期、突然加班或天气变化。
	4. detail_clues 给出 2–4 个群成员顺手会提到的具体细节，不要凑成时间、地点、人数、分工四件套。
		5. final_result 记录聊天停下时大家知道的状态；本条必须形成恰好一个未来仍需执行的明确约定，
		   但最后一条不必把它复述成会议纪要。
6. group_name 是群成员长期会使用的自然群名，2–12 个字符；优先来自成员关系、固定场域、共同爱好、
   内部称呼或轻微玩笑，不能只是本次活动名称。不要默认套用“周末小分队”“周末搭子群”“XX小分队”
   “XX行动组”“周末去哪儿”等万能群名，也不要包含括号中的人数。
7. 事件低风险，不含真实隐私；只有服务评价或证言场景才标明“模拟案例”，普通生活场景的
   topic 和消息中不要出现“模拟案例”。
	8. 本条不能只是围观、闲聊、回忆、投票未决或处理已经结束的即时小事。约定不必总是聚会或改期，
	   但必须能明确回答“绿色用户做什么、哪天、几点”；task 禁止使用“处理事情”等空泛表述。
	9. participants[0] 是截图右侧绿色气泡用户，也是约定的 owner。confirmed_schedule 字段固定为
	   owner, task, date, time, note；owner 必须逐字使用 participants[0].name；date 为 YYYY-MM-DD，
	   time 为 HH:MM；note 只放地点、携带物或条件，没有则为空字符串。今天是 {generation_date}，
	   约定必须晚于聊天发生时间。
	10. 日期、星期和 confirmed_schedule 必须符合真实公历；不确定星期时只写日期，不要猜“周几”。

	输出字段固定为 group_name, group_relationship, common_task, participants, temporary_conflict, detail_clues,
	final_result, confirmed_schedule, safety_note。""",
        temperature=1.0,
    )
    brief_names, _ = dataset_generation.group_brief_contract(brief)
    confirmed_schedule = group_schedule_brief_contract(brief, brief_names)
    brief_focus = dataset_generation.compact_json(
        {key: brief.get(key) for key in ("group_name", "common_task")}
    )
    dataset_generation.reject_forbidden_brief_domains(brief_focus, direction)
    brief_roster_instruction = (
        f"唯一成员姓名白名单：{dataset_generation.compact_json(brief_names)}。"
        "beats.speaker 只能逐字复制其中一个完整姓名；不得输出 participants，"
        "不得新增成员、改名、使用昵称、简称或称谓。"
    )

    beat_plan = client.complete(
        "你是微信群聊结构编辑。你只规划人物互动和消息节拍，不写完整台词，只输出合法 JSON。",
        f"""上一步的群聊场景蓝图：{dataset_generation.compact_json(brief)}
强制身份约束：{brief_roster_instruction}

	请将蓝图展开为恰好 {message_count} 个消息节拍，像截取真实群里的一小段消息，而不是主持人带着
	所有成员依次完成议程。群聊可以从回复、引用、照片后的反应或半截上下文开始；没有必要的小插曲
	就直接自然推进，不要强造冲突。
	最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
事实优先级固定为 final_result > temporary_conflict > 初始提议。temporary_conflict 中的日期、
时间、地点或方案只是待解决的旧状态；解决后所有 beat 必须与 final_result 一致，不能再切回旧状态。

规划时必须满足：
		- 至少 3 人实际发言，人物在群里的位置和说话风格明显不同；除非事情确实需要，禁止安排成员分工汇报。
- 不严格按 A-B-C-A 轮流；允许同一人连续发两条、两个人短暂聊起来，或潜水成员只说一句关键话。
		- 整段安排 3–5 条 2–7 个字的短回应，并允许 2–3 条只表达反应、打趣或承接、不增加任务信息的消息；
		  其余消息要带足上下文，不能把所有人的话都压成电报体。
- 可以有消息交叉、延迟回应、自然称呼或熟人间的省略，但不要为了技巧牺牲可读性。
		- 信息逐步出现，至少两个 detail_clues 分散进入不同消息；至少 3 条消息只是在回应、追问、接梗或补充感受，
		  不得每条都推进安排。
			- final_summary 记录 confirmed_schedule 已经成立；不要求某个成员逐字总结。
	- 必须给 participants[0] 安排足够的绿色消息节拍：至少一个 beat 明确承担或接受 task，并由其
	  绿色气泡直接说出 date 对应的“M月D日”和自然但无歧义的钟点。消息里优先规划“晚上七点半”
	  “下午3点”这类口语，禁止把“M月D日HH:MM”拼成日历字段。日期、时间和任务可以分布
	  在相邻两条绿色消息中，但不能只由其他成员的白色消息提供，也不能只写“我也去”“时间不变”。
	  任务用当事人会说的短动作，不照抄 confirmed_schedule.task 的完整书面标题；note 非空时也必须进入绿色消息节拍。
	- 最后一个 beat 使用符合该人物习惯的短动作或回应自然收尾，不得照抄固定结束语；只要不推翻
	  final_summary，也不得以新问题、新建议或待确认事项结尾。
- 可规划 0–1 个必要的“[图片]”“[文件]”“[引用]”“[撤回]”或“[表情]”提示。
- beats.speaker 必须逐字使用姓名白名单中的完整姓名。
- 普通文本的 message_form 必须写 plain_text；不要写“文字”或“文本”。整段最多一个 beat
  使用真实媒体提示，其余全部为 plain_text。

beats 每项字段固定为 speaker, stage, reply_to, key_detail, message_form；
输出字段固定为 beats, speaking_order_note, final_summary。不得重复输出 participants。""",
        temperature=0.75,
    )
    locked_plan = dict(beat_plan)
    locked_plan["participants"] = brief["participants"]
    plan_names, plan_speakers, _ = dataset_generation.group_plan_contract(
        locked_plan
    )
    roster_instruction = (
        f"成员姓名白名单：{dataset_generation.compact_json(plan_names)}。"
        f"逐条 speaker 顺序：{dataset_generation.compact_json(plan_speakers)}。"
        "participants、messages.speaker 和 schedule.owner 只能使用姓名白名单中的完整姓名；"
        "不得新增成员、昵称、简称或称谓。"
    )

    draft = client.complete(
        "你是中文微信群聊作者。你把既定场景蓝图写成自然群聊，只输出合法 JSON，不写旁白。",
        f"""群聊场景蓝图：{dataset_generation.compact_json(brief)}
	人物与互动节拍：{dataset_generation.compact_json(locked_plan)}
	最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
	强制身份约束：{roster_instruction}

严格沿用人物关系、眼前事项、成员状态和最终事实，把每个 beat 写成一条消息。
蓝图的 final_result 和节拍的 final_summary 是唯一最终事实；冲突中的旧日期、旧地点或旧方案只能
出现在解决之前，解决之后不得再次使用。聊天结束时不能违背 final_summary，但最后一条不必复述它。
	不要严格轮流发言，不要让每个人都说完整长句，不要在第一条一次交代完全部信息，也不要让成员
	逐个回复“收到”“没问题”“我可以”。允许同一人连发、接错话、只回一个短词或顺手插一句生活话。
		整段只安排 3–5 条 2–7 个字的短消息，其余多数为 8–18 个字，硬上限 24 个字；短句用来反应、
		承接或打趣，不能把本可一次说清的话机械拆成电报体。信息较多时
分散到相邻 beat，必须保持语义完整，禁止靠省略号截断半句话。
	称呼和指代要像熟人群聊；“好的、收到、没问题、辛苦了、期待、安排上、确认一下”不能成为机械
	回执。不要让每个人复述前文日期、地点或分工。媒介提示只能作为 text 的简短前缀。
		除正式工作通知外，不要写“X人同行，确认”“分工明确”“方案通过”“大家一致同意”这类会议记录句。
		不要默认把聊天写成“有人发通知—大家报状态—主持人总结”；至少 3 条消息可以只是追问、接梗、
		迟到的回应或和熟人有关的小插曲，不新增任务事实。
	整段最多只有一条消息可以同时完整写出日期、时间、地点或多人分工，其他人用上下文能懂的简称和指代。
	不要把 schedule.task 当作台词照抄。前文已说清对象时，绿色方只需说“我去拿”“我带过去”
	或“我晚点发群里”。日期与时间写成自然语序，例如“8月25日晚上七点半我过去”，禁止
	“8月25日19:30领取物品”这种标签式句子。整段最多明确说一次最终日期和钟点。
message_form 为 plain_text 时只输出对话正文，严禁添加“[文字]”或“[文本]”；只有规划中唯一的
媒体 beat 才能保留对应提示。最后一条要自然落地，可以是动作、短回应或顺手提醒，不能再提问或提出新建议。
participants 使用 2–3 个汉字的正式中文姓名；所有 speaker 和 schedule.owner 必须逐字复制该姓名，
不能改用昵称、称谓、简称或代号。
			schedule 必须恰好 1 项，沿用 confirmed_schedule 的 date, owner, task, time，不得输出 note，
		也不得改成空数组。共同活动只标记一次，不要按参与者复制成多项。
		participants[0] 的绿色气泡必须亲自承担或接受该任务，并清楚出现“M月D日”和自然但无歧义的钟点；
		可以拆成相邻两条绿色消息以保持自然，但不能只靠其他成员补全。note 非空时必须在绿色气泡中说出。
		旧方案被替代后，绿色方必须重新说清最终日期和时间。
group_name 必须沿用蓝图中的自然群名；topic 用来概括这次聊天内容，二者不能混用。禁止把群名
擅自改回“周末小分队”“周末搭子群”或其他“时间词 + 万能组织名”的模板。
conversation_id 暂用 conv_group_draft_{sequence:04d}。

{dataset_generation.schema_instruction("group")}""",
        temperature=0.8,
    )

    reviewed = client.complete(
        "你是微信群聊质检编辑。依据原始蓝图和互动规划直接修订草稿，只输出合法 JSON。",
        f"""原始群聊场景蓝图：{dataset_generation.compact_json(brief)}
	原始人物与互动节拍：{dataset_generation.compact_json(locked_plan)}
	待检查草稿：{dataset_generation.compact_json(draft)}
	最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
	强制身份约束：{roster_instruction}

	首要任务是去掉“群体会议记录”和统一客服腔；需要时可以重写 text，但不得改 speaker 顺序、事实和安排。
	不要把口语润色成统一、完整、正式的书面句。保留人物各自的短句、停顿、承接词、简短回应、
	上下文省略和不同句长；不要为了工整而让所有人轮流发言。

逐项检查并直接修订：
	1. 能看出群聊关系、眼前话题、人物差异和聊天停下时的状态；不强求分工、冲突或共同结论。
2. 至少 3 人发言，语气与作用不同；不能像一个人换头像，也不能机械轮流。
3. 信息逐步出现，至少两个具体细节分散在不同消息中。
4. 有小问题时自然解决；没有小问题时不得补造。最终状态前后一致，但最后一条不必完整总结。
5. 简短回复、交叉或延迟只在自然时保留；每条适合单屏气泡。
6. group_name 像长期存在的真实群名且不是事件摘要；避免“周末、搭子、小分队、行动组”的模板化组合，
   topic 才负责概括本次聊天。
7. 所有正式姓名必须是 2–3 个汉字；speaker 和 schedule.owner 必须与 participants 姓名逐字一致，禁止昵称或称谓。
8. 严格修正人数、长度、字段、speaker、时间和 schedule。
9. 不得增加、删除或合并消息，messages 必须保持恰好 {message_count} 条；日期必须是真实公历日期。
10. 删除所有“[文字]”“[文本]”技术标记；最后一条不得以问号、建议或待确认为结尾，但也不要强行
    写成“最终确定如下”的总结。
11. 逐条检查 text 不超过 24 个字符；超长时改写成意思完整的自然短句，禁止截断词语。
	12. 对照 final_result、final_summary 和 schedule 检查日期、时间、地点与分工；冲突中的旧方案
	不得出现在解决后的消息或最终确认中。
			    schedule 必须恰好 1 项，并与 confirmed_schedule 的 date、owner、time 一致，task 仅保留最终正文支持的核心活动与细节；
			    owner 必须是 participants[0]，禁止给每位成员各复制一项。
13. 用真实公历核对“X月X日周几”和 schedule；无法确定时删除星期，只保留数字日期。
	14. 若成员逐个汇报、逐个确认、频繁复述前文，或交换头像后语气没有区别，改成不对称的真实群聊；
		    整段保留 3–5 条 2–7 个字的短回应，其余多数消息补足到 8–18 个字，避免全篇电报体；
		    同时保留少量无任务信息但有人情味的承接句。
		15. 删除“X人同行，确认”“分工明确”“方案通过”“大家一致同意”等会议记录式表达；正式通知群也要
		    让回应像具体的人，而不是状态回执机器人。
	16. 逐条确认绿色气泡能直接读到最终任务、M月D日和自然但无歧义的钟点，不能让白色气泡成为唯一信息来源，
		    也不能使用“时间不变”“我也去”等模糊代称；note 非空时只保留蓝图已有内容，禁止编造。
	17. 若绿色台词像“M月D日HH:MM + 完整任务标题”的日历记录，改成“8月25日晚上七点半我过去”
	    这类真实语序；schedule.time 仍保持 HH:MM。绿色方不要重复前文已经说清的长任务名称，最终日期
	    和钟点整段只明确说一次。若结尾只是“稳了、妥了、记下了、收到、到时见”，改成承接群内
	    某个具体细节的反应，或让聊天自然停在一条具体提醒上。

{dataset_generation.schema_instruction("group")}""",
        temperature=0.35,
    )
    reviewed["conversation_id"] = (
        f"conv_group_{datetime.datetime.now():%Y%m%d_%H%M%S}_{sequence:04d}"
    )
    return dataset_generation.validate_with_repair(
        client, reviewed, "group", group_plan=locked_plan
    )


if __name__ == "__main__":
    raise SystemExit(
        dataset_generation.dataset_cli("group", generate_group_one)
    )
