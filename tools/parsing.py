import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)


def strip_think_blocks(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


def _parse_tool_call_body(raw):
    raw = raw.strip()

    try:
        call = json.loads(raw)
        if "name" in call:
            return call
    except (json.JSONDecodeError, ValueError):
        pass

    name_m = re.search(r'<name>\s*(.*?)\s*</name>', raw, re.DOTALL)
    args_m = re.search(r'<arguments>\s*(.*?)\s*</arguments>', raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}

    return None


def extract_tool_calls(content):
    cleaned = strip_think_blocks(content)

    cleaned = re.sub(
        r'```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```',
        r'\1',
        cleaned,
        flags=re.DOTALL
    )

    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, cleaned, re.DOTALL)

    if not matches:
        logger.debug("No <tool_call> tags found in content (%d chars): %s",
                      len(content), content[:500])
        return None, content

    logger.debug("Found %d <tool_call> match(es)", len(matches))

    tool_calls = []
    for i, match in enumerate(matches):
        call = _parse_tool_call_body(match)
        if call:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments", {}),
                        ensure_ascii=False
                    )
                }
            })
        else:
            logger.warning("Failed to parse tool_call[%d] — raw: %s", i, match[:300])
            continue

    if not tool_calls:
        return None, content

    remaining = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL).strip()
    return tool_calls, remaining or None


def _tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0
