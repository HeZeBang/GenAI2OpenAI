import json


TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

When you need to call a tool, you MUST use the following XML format. Do NOT use markdown code blocks.

<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. You can call multiple tools by using multiple <tool_call> blocks.
2. If you don't need any tool, just respond normally in plain text without any <tool_call> tags.
3. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
4. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
5. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""

TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" in your response.'
)


def format_tool_definitions(tools):
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f"<tool_definition>\n"
            f"  <name>{func['name']}</name>\n"
            f"  <description>{func.get('description', '')}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool_definition>"
        )
    return "\n".join(definitions)


def inject_tool_prompt(messages, tools, tool_choice=None):
    tool_defs = format_tool_definitions(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

    new_messages = []
    has_system = False

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            new_messages.append(
                {
                    "role": "system",
                    "content": msg.get("content", "") + "\n\n" + tool_prompt,
                }
            )
            has_system = True

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<tool_result>\n"
                        f"  <tool_call_id>{tool_call_id}</tool_call_id>\n"
                        f"  <result>\n{msg.get('content', '')}\n  </result>\n"
                        f"</tool_result>"
                    ),
                }
            )

        elif role == "assistant" and msg.get("tool_calls"):
            tc_text = msg.get("content") or ""
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                call_obj = {
                    "name": func.get("name", ""),
                    "arguments": json.loads(func.get("arguments", "{}")),
                }
                tc_text += f"\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>"
            new_messages.append({"role": "assistant", "content": tc_text.strip()})

        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages
