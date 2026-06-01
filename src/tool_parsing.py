"""
tool_parsing.py

Regex-based parsing of tool invocations from LLM response text.
Supports fenced code blocks, [TOOL_CALL] blocks, and XML-style <invoke> blocks.
"""

import re
import json
import logging
from typing import List, Optional

from src.agent_tools import ToolBlock, TOOL_TAGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Pattern 1: ```bash ... ``` fenced code blocks
_TOOL_BLOCK_RE = re.compile(
    r"```(" + "|".join(TOOL_TAGS) + r")\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

# Pattern 2: [TOOL_CALL] ... [/TOOL_CALL] blocks (some models use this format)
# Matches: {tool => "shell", args => {--command "ls -la"}} etc.
_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALL\]\s*\{([\s\S]*?)\}\s*\[/TOOL_CALL\]",
    re.IGNORECASE,
)

# Pattern 3: XML-style tool calls (minimax, some other models)
# <minimax:tool_call><invoke name="bash"><parameter name="command">...</parameter></invoke></minimax:tool_call>
# Also handles: <tool_call><invoke ...>, <function_call><invoke ...>, plain <invoke ...>
_XML_TOOL_CALL_RE = re.compile(
    r"<(?:[\w]+:)?(?:tool_call|function_call)>\s*([\s\S]*?)</(?:[\w]+:)?(?:tool_call|function_call)>",
    re.IGNORECASE,
)
_XML_INVOKE_RE = re.compile(
    r'<invoke\s+name=["\'](\w+)["\']>\s*([\s\S]*?)</invoke>',
    re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r'<parameter\s+name=["\'](\w+)["\']>([\s\S]*?)</parameter>',
    re.IGNORECASE,
)

# Pattern 4: <tool_code> blocks (MiniMax-M2.5 style)
# {tool => 'tool_name', args => '<param>value</param>'}
_TOOL_CODE_RE = re.compile(
    r"<tool_code>\s*\{([\s\S]*?)\}\s*</tool_code>",
    re.IGNORECASE,
)

# Pattern 5: DeepSeek DSML markup leaking into content. When deepseek
# models can't emit structured tool_calls (e.g. we sent no tool schemas
# that round, or the API didn't parse them), they fall back to raw
# markup using fullwidth-pipe delimiters:
#   <｜｜DSML｜｜tool_calls>
#     <｜｜DSML｜｜invoke name="web_search">
#       <｜｜DSML｜｜parameter name="query" string="true">QUERY</｜｜DSML｜｜parameter>
#     </｜｜DSML｜｜invoke>
#   </｜｜DSML｜｜tool_calls>
# We normalize it into the standard <invoke>/<parameter> form so the
# existing XML parser + stripper handle it (parse → execute; strip →
# never show the garbage to the user). The pipe run is tolerant of
# fullwidth (U+FF5C) and ascii '|' in any count.
_DSML_PIPES = r"[｜|]+"
def _normalize_dsml(text: str) -> str:
    if "DSML" not in text:
        return text
    t = text
    t = re.sub(rf"<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*tool_calls\s*>", "<tool_call>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*tool_calls\s*>", "</tool_call>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*invoke\s+name=", "<invoke name=", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*invoke\s*>", "</invoke>", t, flags=re.IGNORECASE)
    # parameter open tag — drop any extra attrs (e.g. string="true").
    t = re.sub(rf'<\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*parameter\s+name=(["\'][^"\']+["\'])[^>]*>',
               r"<parameter name=\1>", t, flags=re.IGNORECASE)
    t = re.sub(rf"<\s*/\s*{_DSML_PIPES}\s*DSML\s*{_DSML_PIPES}\s*parameter\s*>", "</parameter>", t, flags=re.IGNORECASE)
    return t

# Map model tool names to our tool types
_TOOL_NAME_MAP = {
    "shell": "bash",
    "bash": "bash",
    "terminal": "bash",
    "command": "bash",
    "execute": "bash",
    "run": "bash",
    "python": "python",
    "code": "python",
    "search": "web_search",
    "web_search": "web_search",
    "websearch": "web_search",
    "read": "read_file",
    "read_file": "read_file",
    "cat": "read_file",
    "write": "write_file",
    "write_file": "write_file",
    "save": "write_file",
    "document": "update_document",
    "update_document": "update_document",
    "create_document": "create_document",
    "edit": "edit_document",
    "edit_document": "edit_document",
    "search_chats": "search_chats",
    "search_conversations": "search_chats",
    "find_chat": "search_chats",
    "chat_with_model": "chat_with_model",
    "ask_model": "chat_with_model",
    "chat_model": "chat_with_model",
    "create_session": "create_session",
    "new_session": "create_session",
    "list_sessions": "list_sessions",
    "send_to_session": "send_to_session",
    "message_session": "send_to_session",
    "pipeline": "pipeline",
    "chain": "pipeline",
    "manage_session": "manage_session",
    "session_control": "manage_session",
    "manage_memory": "manage_memory",
    "memory": "manage_memory",
    "manage_tasks": "manage_tasks",
    "tasks": "manage_tasks",
    "schedule": "manage_tasks",
    "list_models": "list_models",
    "models": "list_models",
    "available_models": "list_models",
    "ui_control": "ui_control",
    "ui": "ui_control",
    "control": "ui_control",
    "api_call": "api_call",
    "api": "api_call",
    "integration": "api_call",
    "ask_teacher": "ask_teacher",
    "teacher": "ask_teacher",
    "manage_skills": "manage_skills",
    "skills": "manage_skills",
    "skill": "manage_skills",
    "suggest_document": "suggest_document",
    "suggest": "suggest_document",
    "review_document": "suggest_document",
    "manage_endpoints": "manage_endpoints",
    "endpoints": "manage_endpoints",
    "manage_mcp": "manage_mcp",
    "mcp_servers": "manage_mcp",
    "manage_webhooks": "manage_webhooks",
    "webhooks": "manage_webhooks",
    "manage_tokens": "manage_tokens",
    "tokens": "manage_tokens",
    "manage_documents": "manage_documents",
    "documents": "manage_documents",
    "manage_research": "manage_research",
    "list_research": "manage_research",
    "read_research": "manage_research",
    "open_research": "manage_research",
    "delete_research": "manage_research",
    "manage_settings": "manage_settings",
    "settings": "manage_settings",
    "preferences": "manage_settings",
    "manage_notes": "manage_notes",
    "notes": "manage_notes",
    "todo": "manage_notes",
    "todos": "manage_notes",
}


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def _parse_tool_call_block(raw: str) -> Optional[ToolBlock]:
    """Parse a [TOOL_CALL] block into a ToolBlock.

    Handles formats like:
      {tool => "shell", args => {--command "ls -la"}}
      {tool: "shell", command: "ls -la"}
    """
    # Try to extract tool name
    tool_match = re.search(r'tool\s*(?:=>|:|=)\s*["\']?(\w+)["\']?', raw, re.IGNORECASE)
    if not tool_match:
        return None

    tool_name = tool_match.group(1).lower()
    # Fall back to the raw name when it's a real tool but not in the alias
    # map, so known tools (e.g. manage_calendar) aren't silently dropped.
    mapped = _TOOL_NAME_MAP.get(tool_name) or (tool_name if tool_name in TOOL_TAGS else None)
    if not mapped:
        return None

    # Extract the command/content — try several patterns
    content = None

    # Pattern: --command "value" or --command 'value'
    cmd_match = re.search(r'--command\s+["\'](.+?)["\']', raw, re.DOTALL)
    if cmd_match:
        content = cmd_match.group(1)

    # Pattern: command => "value" or command: "value"
    if not content:
        cmd_match = re.search(r'command\s*(?:=>|:|=)\s*["\'](.+?)["\']', raw, re.DOTALL)
        if cmd_match:
            content = cmd_match.group(1)

    # Pattern: args => {content} — extract everything inside the nested braces
    if not content:
        args_match = re.search(r'args\s*(?:=>|:|=)\s*\{([\s\S]*)\}', raw, re.DOTALL)
        if args_match:
            inner = args_match.group(1).strip()
            # Strip quotes and key prefixes
            inner = re.sub(r'^--?\w+\s+', '', inner)
            inner = inner.strip('\'"')
            if inner:
                content = inner

    # Pattern: query/path/code => "value"
    if not content:
        for key in ("query", "path", "code", "content", "text", "file"):
            m = re.search(rf'{key}\s*(?:=>|:|=)\s*["\'](.+?)["\']', raw, re.DOTALL)
            if m:
                content = m.group(1)
                break

    # Last resort: take everything after the tool declaration
    if not content:
        rest = raw[tool_match.end():].strip()
        rest = re.sub(r'^[,;]\s*', '', rest)
        rest = rest.strip('{} \t\n\'"')
        if rest:
            content = rest

    if content:
        return ToolBlock(mapped, content.strip())
    return None


def _parse_xml_invoke(inv_match) -> Optional[ToolBlock]:
    """Parse an <invoke name="tool"><parameter ...>...</parameter></invoke> match.

    Delegates content-shaping to function_call_to_tool_block — the SAME
    converter used for native function calls — so the full tool set (every
    name in TOOL_TAGS, plus email + MCP tools) and the correct per-tool
    content format are handled in ONE place. The previous version duplicated
    a partial, hand-maintained tool-name map plus a `key: value` serializer:
    any tool missing from that map (e.g. `manage_calendar`) was silently
    dropped, and JSON-arg tools got an unparseable `k: v` blob. Both bugs
    made deepseek's DSML `create_event` calls vanish with no execution.
    """
    # Lowercase the tool name: models often emit capitalized invoke names
    # (e.g. <invoke name="Bash">) and function_call_to_tool_block matches
    # case-sensitively against the lowercase _TOOL_NAME_MAP / TOOL_TAGS, so a
    # raw capitalized name would be silently dropped.
    tool_name = inv_match.group(1).lower()
    body = inv_match.group(2)
    params = {}
    for pm in _XML_PARAM_RE.finditer(body):
        params[pm.group(1)] = pm.group(2).strip()
    # Local import to avoid a circular import at module load.
    from src.tool_schemas import function_call_to_tool_block
    return function_call_to_tool_block(tool_name, json.dumps(params))


def _parse_tool_code_block(raw: str) -> Optional[ToolBlock]:
    """Parse a <tool_code>{tool => 'name', args => '...'}</tool_code> block (MiniMax style)."""
    # Extract tool name
    tool_match = re.search(r"tool\s*=>\s*['\"](\S+?)['\"]", raw)
    if not tool_match:
        return None
    tool_name = tool_match.group(1).lower().replace('-', '_')
    # Strip MCP prefixes like "mcp__server__" or "cli-mcp-server-"
    for prefix in ("mcp__", "cli_mcp_server_", "desktop_commander_", "mcp_code_executor_"):
        if tool_name.startswith(prefix):
            tool_name = tool_name[len(prefix):]
            break

    mapped = _TOOL_NAME_MAP.get(tool_name)

    # Extract args content
    args_match = re.search(r"args\s*=>\s*['\"]?\s*([\s\S]*?)\s*['\"]?\s*$", raw, re.DOTALL)
    args_body = args_match.group(1).strip().strip("'\"") if args_match else ""

    # Parse XML params inside args (e.g. <command>ls</command>)
    xml_params = {}
    for pm in re.finditer(r"<(\w+)>([\s\S]*?)</\1>", args_body):
        xml_params[pm.group(1)] = pm.group(2).strip()

    # When the model gave structured params, hand them to the canonical
    # converter (same as native calls + <invoke>) so the full tool set and
    # correct per-tool content format apply — not a partial map + k:v blob.
    if xml_params:
        from src.tool_schemas import function_call_to_tool_block
        block = function_call_to_tool_block(mapped or tool_name, json.dumps(xml_params))
        if block:
            return block

    # No structured params: args_body is a raw single value (e.g. a bash
    # command). Keep the freeform special-casing for the simple tools.
    if mapped:
        if mapped == "bash":
            content = xml_params.get("command", args_body)
        elif mapped == "python":
            content = xml_params.get("code", args_body)
        elif mapped == "web_search":
            content = xml_params.get("query", args_body)
        elif mapped in ("read_file", "write_file"):
            content = xml_params.get("path", xml_params.get("file_path", args_body))
        else:
            content = "\n".join(f"{k}: {v}" for k, v in xml_params.items()) if xml_params else args_body
        if content:
            return ToolBlock(mapped, content.strip())
    elif tool_name and args_body:
        # Unknown tool — try as MCP tool call
        content = "\n".join(f"{k}: {v}" for k, v in xml_params.items()) if xml_params else args_body
        return ToolBlock(tool_name, content.strip())
    return None


def parse_tool_blocks(text: str) -> List[ToolBlock]:
    """Extract executable tool blocks from LLM response text.

    Supports multiple formats:
    1. ```bash ... ``` fenced code blocks (standard)
    2. [TOOL_CALL] ... [/TOOL_CALL] blocks (some models)
    3. XML-style <tool_call>/<invoke> blocks
    4. <tool_code> blocks (MiniMax-M2.5 style)
    5. DeepSeek DSML markup (normalized to <invoke> first)
    """
    blocks = []

    # Normalize DeepSeek DSML markup into standard <invoke> form so the
    # XML patterns below catch it.
    text = _normalize_dsml(text)

    # Pattern 1: fenced code blocks
    for m in _TOOL_BLOCK_RE.finditer(text):
        tag = m.group(1).lower()
        content = m.group(2).strip()
        if not content:
            continue
        # If a code block's content is an <invoke> XML call (some models wrap
        # tool calls in ```python or ```xml fences), parse the invoke instead.
        if '<invoke' in content:
            invoked = False
            for inv in _XML_INVOKE_RE.finditer(content):
                block = _parse_xml_invoke(inv)
                if block:
                    blocks.append(block)
                    invoked = True
            if invoked:
                continue
        blocks.append(ToolBlock(tag, content))

    # Pattern 2: [TOOL_CALL] blocks (only if no fenced blocks found)
    if not blocks:
        for m in _TOOL_CALL_RE.finditer(text):
            block = _parse_tool_call_block(m.group(1))
            if block:
                blocks.append(block)

    # Pattern 3: XML-style <tool_call>/<invoke> blocks
    if not blocks:
        # Try wrapped: <tool_call><invoke ...>...</invoke></tool_call>
        for m in _XML_TOOL_CALL_RE.finditer(text):
            for inv in _XML_INVOKE_RE.finditer(m.group(1)):
                block = _parse_xml_invoke(inv)
                if block:
                    blocks.append(block)
        # Try bare <invoke> without wrapper
        if not blocks:
            for inv in _XML_INVOKE_RE.finditer(text):
                block = _parse_xml_invoke(inv)
                if block:
                    blocks.append(block)

    # Pattern 4: <tool_code> blocks (MiniMax-M2.5 style)
    if not blocks:
        for m in _TOOL_CODE_RE.finditer(text):
            block = _parse_tool_code_block(m.group(1))
            if block:
                blocks.append(block)

    # Pattern 5: Gemma native tool-call bridge. Gated — only fires when nothing
    # else matched AND the text carries Gemma's specific call signature, so it's
    # a no-op for every other model/format.
    if not blocks:
        blocks.extend(_parse_gemma_tool_calls(text))

    return blocks


# Gemma-family models emit tool calls in their own non-standard format that none
# of the parsers above recognize, e.g.:
#   ◁tool_call▷call:google:search{queries:[◁"▷q◁"▷]}◁/tool_call▷
#   <|tool_call>call:search(query="...")
# The delimiter renders differently by tokenizer (◁ ▷ glyphs, <| |>, < >), so we
# allow spaces / pipes / slashes around the word. We don't execute these; we just
# strip them (open token → close token, or to end-of-string if unclosed).
_GEMMA_TOOL_RE = re.compile(
    r'[◁<][\s|/]*tool_call[\s|/]*[▷>][\s\S]*?(?:[◁<][\s|/]*tool_call[\s|/]*[▷>]|$)',
    re.IGNORECASE,
)
# Leftover bare Gemma special-token glyphs (e.g. ◁"▷, ◁end_of_turn▷).
_GEMMA_STRAY_TOKEN_RE = re.compile(r'◁[^▷]{0,40}?▷')


# ── Gemma tool-call bridge (agent mode) ──
# Gemma-4 / 3n were trained on a Google-style tool API and emit, e.g.:
#   <|tool_call>call:google:search{queries:[<|"|>q1<|"|>, <|"|>q2<|"|>]}<tool_call|>
#   ◁tool_call▷call:search{queries:[◁"▷q◁"▷]}◁/tool_call▷
#   <|tool_call>call:search(query="...")
# none of which the parsers above recognize. We bridge ONLY the calls we can
# safely map to a real Odysseus tool (search -> web_search). _GEMMA_CALL_RE is
# the gate: it matches only Gemma's signature, so this never affects other
# models, and the token cleanup is scoped to the matched call's args.
_GEMMA_CALL_RE = re.compile(
    r'[◁<][\s|/]*tool_call[\s|/]*[▷>]\s*call\s*:\s*([\w:.\-]+)\s*([{(][\s\S]*?[})])',
    re.IGNORECASE,
)
# Gemma's trained tool names -> Odysseus tools. Only search is safely bridgeable;
# other Gemma calls are left unparsed (we don't fabricate calls it can't make).
_GEMMA_TOOL_NAME_MAP = {
    "google:search": "web_search", "google_search": "web_search",
    "search": "web_search", "web_search": "web_search", "websearch": "web_search",
}
# Gemma wraps string literals in special-token glyphs (<|"|> or ◁"▷); normalize
# them to a plain double-quote so the query strings can be read.
_GEMMA_QUOTE_RE = re.compile(r'[◁<][\s|/]*"[\s|/]*[▷>]')


def _parse_gemma_tool_calls(text: str) -> List["ToolBlock"]:
    """Bridge Gemma's native tool-call format to Odysseus tool blocks.

    Returns [] unless the text carries Gemma's specific call signature, so it is
    a no-op for every other model. Currently maps only search -> web_search.
    """
    blocks: List[ToolBlock] = []
    for m in _GEMMA_CALL_RE.finditer(text):
        name = (m.group(1) or "").strip().lower()
        mapped = _GEMMA_TOOL_NAME_MAP.get(name)
        if mapped != "web_search":
            continue  # only search is safe to bridge; don't fabricate others
        args = _GEMMA_QUOTE_RE.sub('"', m.group(2) or "")  # scoped token cleanup
        # queries:[ "a", "b" ]  (list form)
        qlist = re.search(r'quer(?:y|ies)\s*[:=]\s*\[([\s\S]*?)\]', args, re.IGNORECASE)
        if qlist:
            queries = re.findall(r'["\']([^"\']+)["\']', qlist.group(1))
        else:
            # query="a"  (single/paren form), or bare query=a
            single = re.search(r'quer(?:y|ies)\s*[:=]\s*["\']([^"\']+)["\']', args, re.IGNORECASE)
            if single:
                queries = [single.group(1)]
            else:
                bare = re.search(r'quer(?:y|ies)\s*[:=]\s*([^,)}\]]+)', args, re.IGNORECASE)
                queries = [bare.group(1)] if bare else []
        for q in queries[:3]:
            q = q.strip().strip('"\',')
            if q:
                blocks.append(ToolBlock("web_search", q))
    return blocks


def strip_tool_blocks(text: str) -> str:
    """Remove executable tool blocks from text for clean display."""
    # Normalize DSML first so its markup gets stripped by the <invoke>
    # / <tool_call> removers below instead of leaking to the user.
    text = _normalize_dsml(text)
    cleaned = _TOOL_BLOCK_RE.sub('', text)
    cleaned = _TOOL_CALL_RE.sub('', cleaned)
    cleaned = _XML_TOOL_CALL_RE.sub('', cleaned)
    cleaned = _TOOL_CODE_RE.sub('', cleaned)
    # Gemma native tool-call tokens + stray special-token glyphs.
    cleaned = _GEMMA_TOOL_RE.sub('', cleaned)
    cleaned = _GEMMA_STRAY_TOKEN_RE.sub('', cleaned)
    # Strip bare <invoke> blocks not wrapped in <tool_call>
    cleaned = re.sub(r'<invoke\s+name=["\'].*?</invoke>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
