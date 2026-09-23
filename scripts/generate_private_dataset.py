#!/usr/bin/env python3
"""Generate one-to-one WeChat datasets through a private-chat prompt chain."""

import datetime

import dataset_generation

PRIVATE_SCENE_FORMULA = "人物关系 + 当下缘由 + 双方状态 + 明确约定 + 自然落点"
CONFIRMED_SCHEDULE_KEYS = ("owner", "task", "date", "time", "note")


def private_schedule_brief_contract(
    brief: dict, participant_names: list[str]
) -> dict:
    """Lock one extraction-ready agreement before later prompt-chain stages."""
    item = brief.get("confirmed_schedule")
    if (
        not isinstance(item, dict)
        or tuple(item.keys()) != CONFIRMED_SCHEDULE_KEYS
    ):
        raise ValueError(
            f"私聊蓝图 confirmed_schedule 字段及顺序必须为 {CONFIRMED_SCHEDULE_KEYS}"
        )
    for key in ("owner", "task", "date", "time"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"私聊蓝图 confirmed_schedule.{key} 不能为空")
    if not isinstance(item.get("note"), str):
        raise ValueError("私聊蓝图 confirmed_schedule.note 必须是字符串")
    if item["owner"] != participant_names[0]:
        raise ValueError(
            "私聊蓝图的明确约定必须由 participants[0]（绿色气泡）负责"
        )
    try:
        datetime.datetime.strptime(item["date"], "%Y-%m-%d")
        datetime.datetime.strptime(item["time"], "%H:%M")
    except ValueError as error:
        raise ValueError(
            "私聊蓝图的明确约定必须使用 YYYY-MM-DD 和 HH:MM"
        ) from error
    return item


def generate_private_one(
    client: dataset_generation.ChatCompletionClient,
    direction: str,
    sequence: int,
) -> dict:
    """Four-link private chain: brief -> beat plan -> draft -> review."""
    message_count = 10 + (sequence - 1) % 3
    generation_date = datetime.datetime.now().date().isoformat()
    brief = client.complete(
        "你是微信私聊场景策划。你只设计故事蓝图，不写对话，只输出合法 JSON。",
        f"""本次开放创作要求：{direction}

		请使用公式“{PRIVATE_SCENE_FORMULA}”原创一个适合单张微信截图、可用于日程抽取训练的私聊场景。
	先自主决定题材、人物关系和事件，不要从预设主题列表中选择，也不要对固定人名、地点、
	物品或句式做随机替换。direction 如果提供了方向，只作为创作意图，不是固定剧情。
	direction 中列出的近期主题是硬性反重复条件，新场景不能只替换食物、地点或活动名称。
	题材不同不等于事情越冷门越好。优先选择真实微信里高频发生的小事，例如顺路取送、接人、
	交接材料、上课前确认、家里的一件事、到店办理或临时帮忙；不要为了显得新颖而设计专业、
	生僻且需要大量解释的任务。人物只聊眼下需要知道的内容，不互相介绍双方早已知道的背景。
		不要习惯性生成“晒烘焙/手作成果后约下次一起做”“东西落在对方家”“周末爬山后吃饭”
		“突然加班所以改期”等合成数据常见套路；除非 direction 明确要求，也不要默认把事情放在周末。

要求：
{dataset_generation.SCHEDULE_GROUNDING_RULES}
1. 关系和交流动机可信，事件普通、低风险，不含真实隐私。场景应像从一段真实生活中截取，
   不要像为展示信息而编写的完整案例；双方已经知道的背景不要重新互相解释。
2. 在 participants 中一次确定两位参与者；每项字段固定为 name, identity, speaking_style。
   name 只能是 2–3 个汉字的正式中文姓名（完整姓名）；禁止“小王”“阿杰”“老李”等昵称，禁止称谓、
   英文名、拼音、数字或“用户A”等代号。后续所有阶段不得改名。
3. detail_clues 给出 2–4 个一致的具体线索，其中至少一个是只有这两个人聊天才会顺手提到的
	   小细节；不要把线索写成任务清单，也不要用“遗落物品”充当万能熟人感。
4. emotional_arc 只记录双方当下情绪如何自然变化；可以几乎不变，禁止为了完整故事强造反转。
	5. everyday_detail 只在它与当下话题或人物状态自然相连时提供；不适合就写“无”，禁止硬塞
	   吃饭、天气、宠物等无关细节来伪装生活感。
6. 只有服务评价或证言场景才说明是“模拟案例”；普通生活场景的 topic 和消息中不要出现
   “模拟案例”。
7. 本条不能只是聊家常、回忆往事、分享感受或处理已经结束的即时小事；聊天结束时必须留下
   恰好一个双方已经说定、未来仍需执行的具体约定。约定不必总是见面或改期，但必须能明确回答
   “谁、做什么、哪天、几点”，task 禁止写成“处理事情”“保持联系”等空泛表述。
8. participants[0] 是截图右侧绿色气泡发送者，也是约定的 owner。confirmed_schedule 字段固定为
   owner, task, date, time, note；owner 必须逐字使用 participants[0].name；date 为 YYYY-MM-DD，
   time 为 HH:MM；note 只放地点、携带物或条件，没有则写空字符串。今天是 {generation_date}，
   约定必须晚于聊天发生时间。
9. 日期、星期和 confirmed_schedule 必须符合真实公历；不确定星期时只写日期，不要猜“周几”。

输出字段固定为 participants, relationship, purpose, event, emotional_arc, detail_clues, everyday_detail,
confirmed_schedule, safety_note。""",
        temperature=1.0,
    )
    brief_names, _ = dataset_generation.private_brief_contract(brief)
    confirmed_schedule = private_schedule_brief_contract(brief, brief_names)
    brief_focus = dataset_generation.compact_json(
        {key: brief.get(key) for key in ("purpose", "event")}
    )
    dataset_generation.reject_forbidden_brief_domains(brief_focus, direction)
    brief_roster_instruction = (
        f"唯一姓名白名单：{dataset_generation.compact_json(brief_names)}。"
        "beats.speaker 只能逐字复制其中一个完整姓名；不得输出 participants，"
        "不得改名或使用昵称、简称、称谓。"
    )

    beat_plan = client.complete(
        "你是私聊故事结构编辑。你只规划人物和消息节拍，不写完整台词，只输出合法 JSON。",
        f"""上一步的五要素蓝图：{dataset_generation.compact_json(brief)}
强制身份约束：{brief_roster_instruction}

请把蓝图展开为恰好 {message_count} 个连续消息节拍，像截取正在发生的一段私聊，而不是写
“提出问题—分析问题—解决问题”的标准剧本。可以从半句话、追问或对上一件事的承接开始。
最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
事实优先级固定为 ending_state > minor_turn > 初始提议；小波折中的旧时间、旧地点或旧方案被解决后，
后续 beat 不得再次当作有效安排。
	至少两个具体细节进入聊天，但不要堆在同一句；everyday_detail 不是“无”且能自然接上时才穿插。
两个人必须都有可辨认的语言习惯，也都必须实际发言。发言数量和句长不要对称；允许一人
		连续发两条，整段安排 3–5 条 2–7 个字的短回应，以及 1–2 条只起承接、反应或拉近关系作用的消息；
		其余消息要带足上下文，不能把整段规划成一句一顿的电报体。
不是每条消息都要增加新事实，不要让双方像客服一样逐项确认。minor_turn 可以很轻，也可以写
“无明显波折”；ending_state 只记录结束时事实，最后一条不必完整复述结论。
必须为 participants[0] 安排足够的绿色消息节拍：至少一个 beat 明确承担 confirmed_schedule.task，
并由其绿色气泡明确说出 confirmed_schedule.date 对应的“M月D日”和自然但无歧义的钟点。
消息里优先规划“晚上七点半”“下午3点”这类口语；不要把“M月D日HH:MM”拼成日历字段。
日期、时间和任务可自然分布在相邻两条绿色消息中，但不能只由 participants[1] 的白色消息提供，
也不能只写“还是那个时间”“到时见”等需要猜测的表达。任务用当事人会说的短动作，不照抄
confirmed_schedule.task 的完整书面标题；note 非空时也必须进入绿色消息节拍。
participants.name 必须是 2–3 个汉字的正式中文姓名；beats.speaker 必须逐字使用对应 name，
禁止昵称、称谓和简称。
可规划 0–1 个必要的“[图片]”“[语音]”“[文件]”“[定位]”或“[引用]”提示。
普通文本的 message_form 必须写 plain_text，不要写“文字”或“文本”；整段最多一个 beat 使用
真实媒体提示，其余全部为 plain_text。

beats 每项字段固定为 speaker, purpose, key_detail, emotion, message_form；
输出字段固定为 beats, minor_turn, ending_state。不得重复输出 participants。""",
        temperature=0.75,
    )
    locked_plan = dict(beat_plan)
    locked_plan["participants"] = brief["participants"]
    plan_names, plan_speakers, _ = dataset_generation.private_plan_contract(
        locked_plan
    )
    roster_instruction = (
        f"成员姓名白名单：{dataset_generation.compact_json(plan_names)}。"
        f"逐条 speaker 顺序：{dataset_generation.compact_json(plan_speakers)}。"
        "participants、messages.speaker 和 schedule.owner 只能使用姓名白名单中的完整姓名；"
        "不得改名或使用昵称、简称、称谓。"
    )

    draft = client.complete(
        "你是中文微信私聊作者。你把既定蓝图写成自然口语，只输出合法 JSON，不写旁白。",
        f"""五要素蓝图：{dataset_generation.compact_json(brief)}
人物与消息节拍：{dataset_generation.compact_json(locked_plan)}
最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
强制身份约束：{roster_instruction}

严格沿用人物、事件、情绪路径和细节，把每个 beat 写成一条消息。
ending_state 是唯一最终事实；minor_turn 中被否定或替代的旧安排只能出现在解决之前，后续消息和
schedule 不得再次使用旧安排。
两个人用词、停顿和句长要有区别；不要每句话都解释背景，不写广告、会议纪要或总结文。
	非约定细节可使用双方都懂的指代和省略，如“那个”“我刚看到”；最终任务、日期和时间不得省略；允许口语碎片，但必须能从
		上下文读懂。整段只安排 3–5 条 2–7 个字的短消息，其余多数为 8–18 个字，硬上限 24 个字；
		短句用于真实反应，不能把本可一次说清的话机械拆成电报体。
信息较多时分散到相邻 beat，禁止靠省略号截断半句话。
		“好的、收到、没问题、辛苦了、期待、安排上、确认一下、说定了”只能在人物关系确实适合时偶尔出现，
		不能连续充当每轮回应。不要让双方重复对方刚说过的日期、地点和任务来证明自己听懂了。
		不要靠“改天请你喝咖啡/吃饭”“等你好消息，希望有惊喜”“你品味真好”等万能客套制造亲近感；
		关系要从两人知道的具体上下文、打断方式和省略中自然显出来。
	禁止用“期待周末”“好期待”“完美”“愉快决定”作结。结尾可以停在短回应、顺手提醒或生活插句，
	不必写“最终确定”“那就这么安排”式总结。
	不要把 schedule.task 当台词照抄。前文已经说清对象时，绿色方只需像真人一样说“我去拿”“我给你送过去”
	或“我上线陪你弄”；不要生硬复述长任务名。日期与时间写成自然语序，例如“8月25日晚上七点半
	我给你打视频”，禁止“8月25日19:30陪你测试设备”这种标签式句子。整段最多明确说一次最终日期和钟点。
participants 使用 2–3 个汉字的正式中文姓名；所有 speaker 和 schedule.owner 必须逐字复制该姓名，
不能改用昵称、称谓、简称或代号。
	媒介提示只能作为 text 的简短前缀。本条 schedule 必须恰好 1 项，沿用 confirmed_schedule 的
date, owner, task, time，不得输出 note，也不得改成空数组。
	participants[0] 的绿色气泡必须亲自确认该任务，并清楚出现“M月D日”和自然但无歧义的钟点；可以拆成
相邻两条绿色消息以保持自然，但不能依赖白色气泡单独提供日期、时间或事项。note 非空时必须在
绿色气泡中自然说出。旧方案被替代后，绿色方必须重新说清最终日期和时间，避免歧义。
	message_form 为 plain_text 时只输出对话正文，严禁添加“[文字]”或“[文本]”；只有规划中唯一的
媒体 beat 才能保留对应提示。
服务评价类的 topic 必须含“模拟案例”。conversation_id 暂用 conv_private_draft_{sequence:04d}。

{dataset_generation.schema_instruction("private")}""",
        temperature=0.8,
    )

    reviewed = client.complete(
        "你是微信私聊质检编辑。依据原始蓝图直接修订草稿，只输出修订后的合法 JSON。",
        f"""原始五要素蓝图：{dataset_generation.compact_json(brief)}
	原始节拍规划：{dataset_generation.compact_json(locked_plan)}
	待检查草稿：{dataset_generation.compact_json(draft)}
	最终明确约定：{dataset_generation.compact_json(confirmed_schedule)}
	强制身份约束：{roster_instruction}

	首要任务是去掉“示范对话”和客服腔；需要时可以重写 text，但不得改 speaker 顺序、事实和安排。
	不要把口语润色成统一、完整、正式的书面句。保留两个人不同的短句、停顿、承接词、简短回应和句长，
	也保留上下文能解释的省略与不对称发言。允许一句话不漂亮、不完整，只要像这个人此刻会发的消息。
检查人物关系、目的、事件、情绪变化和至少两个具体细节是否真正出现在聊天中；
	检查语言习惯是否可区分，小波折是否自然；只有与上下文有关时才保留生活化插句；
不要强行反转，也不要一律生成取消或改期。若每句话都在交代、确认或总结事实，删去书面腔，
改成真实的人会发送的短句；若交换两个人姓名后语气仍完全一样，按 speaking_style 拉开差异。
		整段保留 3–5 条 2–7 个字的自然短回应，其余多数消息补足到 8–18 个字，避免全篇电报体。
		避免连续出现“好的、收到、没问题、辛苦了、期待、安排上”，并删除“随时联系、祝你顺利、
		相信你的潜力、等着看惊喜、不见不散、改天请你吃饭/喝咖啡”等万能客套，改成承接具体上下文的回应。
	并删除“期待周末”“完美”等模型式收尾。逐项检查正式姓名均为 2–3 个汉字，
speaker 和 schedule.owner 与 participants 姓名逐字一致。严格修正人数、长度、字段和时间。
不得增加、删除或合并消息，messages 必须保持恰好 {message_count} 条；日期必须是真实公历日期。
	删除所有“[文字]”“[文本]”技术标记。
		本条是明确约定抽取训练样本：schedule 必须恰好 1 项，并与 confirmed_schedule 的 date、owner、time 一致；
	task 仅保留最终正文能支持的核心活动与细节；owner 必须是 participants[0]。逐条确认绿色气泡中能直接读到最终任务、M月D日和
	自然但无歧义的钟点，不能让白色气泡成为唯一信息来源，也不能使用“时间不变”“还是老地方”等模糊代称。
	如果出现“M月D日HH:MM + 完整任务标题”的日历记录式台词，改成“8月25日晚上七点半我过去”
	这类真实语序；schedule.time 仍保持 HH:MM。绿色方不得重复前文已经清楚的长任务名称，整段最终日期
	和钟点只明确说一次。若最后一条只是“到时见、稳了、记下了、收到”，改成承接具体上下文的反应，
	或让对话自然停在一条具体提醒上。
	note 非空时只保留蓝图已有的地点、携带物或条件，并让绿色气泡明确说出，禁止质检阶段编造。
逐条检查 text 不超过 24 个字符；超长时改写成意思完整的自然短句，禁止截断词语。
用真实公历核对“X月X日周几”和 schedule；无法确定时删除星期，只保留数字日期。

{dataset_generation.schema_instruction("private")}""",
        temperature=0.4,
    )
    reviewed["conversation_id"] = (
        f"conv_private_{datetime.datetime.now():%Y%m%d_%H%M%S}_{sequence:04d}"
    )
    return dataset_generation.validate_with_repair(
        client, reviewed, "private", private_plan=locked_plan
    )


if __name__ == "__main__":
    raise SystemExit(
        dataset_generation.dataset_cli("private", generate_private_one)
    )
