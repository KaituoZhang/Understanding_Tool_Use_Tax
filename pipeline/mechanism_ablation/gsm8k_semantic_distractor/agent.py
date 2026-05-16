import json
import math
import re
import time
from typing import Any, Dict, List, Optional

from .data_loader import Chunk


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def _parse_xml_tool_calls(content: str) -> List[Any]:
    calls = []
    for match in TOOL_CALL_XML_RE.finditer(content or ""):
        try:
            tool_call = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        arguments = tool_call.get("arguments", {})
        calls.append((tool_call.get("name", ""), arguments, json.dumps(arguments, ensure_ascii=False)))
    return calls


def _fix_json_string(text: str) -> str:
    import ast

    try:
        obj = ast.literal_eval(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    fixed = text.replace("True", "true").replace("False", "false").replace("None", "null")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"(?<=[{,\[:\s])'((?:[^'\\]|\\.)*?)'(?=\s*[:,}\]\n])", r'"\1"', fixed)
    return fixed


def _extract_number_fallback(text: str) -> Optional[Dict[str, Any]]:
    for pattern in [
        r"(?:final.?answer|answer|result|total)\s*(?:is|=|:)\s*(\d+(?:\.\d+)?)",
        r"=\s*(\d+(?:\.\d+)?)\s*$",
    ]:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return {"final_answer": match.group(1), "_parse_fallback": True}
    numbers = re.findall(r"(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])", text)
    if numbers:
        return {"final_answer": numbers[-1], "_parse_fallback": True}
    return None


def extract_json(text: str) -> Dict[str, Any]:
    stripped = strip_think_tags(text)
    stripped = re.sub(r"```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"```\s*$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        fallback = _extract_number_fallback(stripped)
        if fallback:
            return fallback
        raise ValueError(f"No JSON object found in output: {stripped[:200]}")

    depth = 0
    end = start
    for index in range(start, len(stripped)):
        if stripped[index] == "{":
            depth += 1
        elif stripped[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    candidate = stripped[start: end + 1]
    for attempt in (candidate, _fix_json_string(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    fallback = _extract_number_fallback(stripped)
    if fallback:
        return fallback
    raise ValueError(f"Could not parse JSON from output: {candidate[:300]}")


CALC_ALLOWED = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "pow": pow,
    "sum": sum,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def safe_calculate_steps(steps: Any, state: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(steps, str):
        steps = [steps]
    if not steps or not isinstance(steps, list):
        return "Error: calc_steps must be a non-empty list"

    environment = dict(CALC_ALLOWED)
    if state:
        for key, value in state.items():
            if re.fullmatch(r"[a-zA-Z_]\w*", str(key)):
                environment[str(key)] = value

    last_result = None
    for raw_step in steps:
        step = str(raw_step).strip()
        if not step:
            continue
        match = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", step)
        if match:
            variable_name, expression = match.group(1), match.group(2).strip()
        else:
            variable_name, expression = None, step

        cleaned = expression
        for name in sorted(environment.keys(), key=len, reverse=True):
            cleaned = cleaned.replace(name, "")
        if re.search(r"[a-zA-Z_]", cleaned):
            return f"Error: disallowed name in step '{step}'"

        try:
            result = eval(expression, {"__builtins__": {}}, environment)
        except Exception as exc:
            return f"Error in step '{step}': {exc}"

        if isinstance(result, float) and math.isfinite(result) and result == int(result):
            result = int(result)
        last_result = result
        if variable_name:
            environment[variable_name] = result
            if state is not None:
                state[variable_name] = result

    return str(last_result) if last_result is not None else "Error: no steps evaluated"


_CALCULATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate arithmetic step by step and return the result of the final step.",
        "parameters": {
            "type": "object",
            "properties": {
                "calc_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sequence of arithmetic steps such as ['step1 = 48/2', 'step2 = 48 + step1'].",
                }
            },
            "required": ["calc_steps"],
        },
    },
}


class LLMBackend:
    def __init__(
        self,
        model: str = "mock-oracle",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        max_attempts: int = 3,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_attempts = max_attempts
        self.api_base = api_base or "mock://local"
        self.api_key = api_key
        self.is_mock = self.api_base.startswith("mock://") or self.model.startswith("mock")
        self.client = None

        if not self.is_mock:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def _call(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        if self.is_mock:
            raise RuntimeError("Mock backends do not support direct LLM calls.")

        last_error = None
        extra_body: Dict[str, Any] = {}
        if "qwen" in self.model.lower():
            extra_body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if json_mode:
            extra_body["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    seed=self.seed,
                    **extra_body,
                )
                return strip_think_tags(response.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    extra_body.pop("response_format", None)
                elif attempt == 1:
                    extra_body.pop("extra_body", None)
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_attempts} attempts: {last_error}")


def _numbered_block(chunks: List[Chunk]) -> str:
    return "\n".join(f"[{chunk.index}] {chunk.text}" for chunk in chunks)


FC_SYSTEM_PROMPT = """\
You are a math-solving agent with a `calculate` function for arithmetic.
You will see a question and numbered information chunks. Some chunks are noise, so identify the real evidence first.

Strategy:
1. Identify which chunks contain real evidence and which chunks are distractors.
2. Extract the quantities that matter.
3. Call the `calculate` function for all arithmetic.
4. After the calculations, return strict JSON with:
{
  "evidence_ids": [integer chunk indices that are real evidence],
  "final_answer": "the numeric answer only",
  "reasoning": "brief explanation of which chunks you used and why"
}

Critical rules:
- "half as many" means divide by 2; "twice as much" means multiply by 2.
- "how much more" means needed minus already-have.
- If the same action applies to N people K times, multiply by both N and K.
- Count every multiplier in the sentence.\
"""
