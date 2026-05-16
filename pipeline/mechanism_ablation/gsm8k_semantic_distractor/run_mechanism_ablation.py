# -*- coding: utf-8 -*-
"""
run_mechanism_ablation.py  —  Tool-Use Mechanism Ablation Study

Decomposes WHY real function-calling (FC) improves performance.
All conditions use the same real OpenAI FC protocol with GPT model.

Research Questions
------------------
  H1 (Evidence Selection): Does the benefit come from having clean input to
     the tool — i.e. the model can correctly identify relevant chunks despite noise?
  H2 (Iterative Refinement): Does the benefit come from multiple tool calls —
     the model retrying and correcting itself (especially for SP noise)?
  H3 (Actual Computation): Does the benefit come from the calculator itself,
     or just from the FC protocol forcing structured reasoning?

Conditions
----------
  fc_baseline   — Standard real FC: real calculator, all noisy chunks, max 5 calls
                  (Should reproduce st_all_fc from run_ablation.py)

  fc_auto       — Same as fc_baseline, but first turn uses tool_choice="auto"
                  instead of forcing tool_choice="required".
                  → Isolates the impact of forced protocol overhead.

  fc_noop       — FC protocol active, but tool echoes expression without computing
                  → If accuracy stays high: benefit is from protocol (structured output),
                    not actual computation.
                  → If accuracy drops heavily: computation is the key driver.  [tests H3]

  fc_auto_noop  — Same as fc_noop, but first turn uses tool_choice="auto"
                  instead of forcing tool_choice="required".
                  → Isolates forced-call overhead under NoOp tools.

  fc_perfect    — FC protocol, tool directly returns the gold answer on every call
                  → Computation upper bound. Shows what's achievable if the calculator
                    always returned the correct answer.  [ceiling test]

  fc_max1       — Real FC, real calculator, but max_tool_calls=1 (no iteration)
                  → Gap vs fc_baseline = value of iterative tool calling.
                    Especially informative for SP-style paraphrase noise.  [tests H2]

  fc_oracle_ev  — Real FC, real calculator, but only evidence chunks shown (no noise)
                  → Gap vs fc_baseline = cost of noise for evidence selection.
                    Answers: "If retrieval were perfect, how much better would FC do?"
                    [tests H1]

Interpreting Results
--------------------
  fc_baseline vs fc_auto:     Gap = effect of forcing first tool call (protocol overhead)
  fc_baseline vs fc_noop:     Large gap → computation crucial; small gap → protocol effect
  fc_noop vs fc_auto_noop:    Gap = pure forced-call overhead when compute is disabled
  fc_baseline vs fc_perfect:  Gap = headroom from imperfect computation/planning
  fc_baseline vs fc_max1:     Gap = value of iteration (per variant)
  fc_baseline vs fc_oracle_ev:Gap = cost of noise / evidence selection failure
  fc_noop vs fc_oracle_ev:    Interaction: does removing noise help even without computation?

Usage
-----
  python -m unified_eval_st.run_mechanism_ablation \\
      --data ../output/train_aug_150_f11b11.jsonl \\
      --fc_model gpt-4.1-mini \\
      --fc_api_base https://api.openai.com/v1 \\
      --fc_api_key $OPENAI_API_KEY \\
      --limit 50 \\
      --save_dir results/mechanism_ablation_f11b11/ \\
      --variants base,TB,PED,HU,SP

  # Run only specific conditions (e.g. to add fc_perfect to existing results):
  python -m unified_eval_st.run_mechanism_ablation \\
      --data ../output/train_aug_150_f11b11.jsonl \\
      --fc_model gpt-4.1-mini \\
      --fc_api_base https://api.openai.com/v1 \\
      --fc_api_key $OPENAI_API_KEY \\
      --limit 50 \\
      --save_dir results/mechanism_ablation_f11b11/ \\
      --only fc_perfect,fc_oracle_ev
"""

import argparse, json, os, re, time
from copy import copy
from typing import Callable, Dict, Any, List, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from .data_loader import load_problems, group_by_question
from .env_singleturn import SingleTurnEnvironment, SingleTurnEnvConfig
from .agent import (
    LLMBackend,
    safe_calculate_steps,
    extract_json,
    strip_think_tags,
    _numbered_block,
    FC_SYSTEM_PROMPT,
    _CALCULATE_TOOL_SCHEMA,
    _parse_xml_tool_calls,
)
from .metrics import (
    answers_equal,
    evidence_precision_recall_f1,
    evidence_exact_match,
    aggregate_results,
    print_report,
)
from .gate.gate_models import GStepGate, GCommitGate


# ---- No-tool CoT prompt (pure chain-of-thought, no FC protocol) ----
FC_NOTOOL_SYSTEM_PROMPT = (
    "You are a careful math solver. Read the numbered information chunks below.\n"
    "Some chunks are noise — use only the relevant ones as evidence.\n"
    "Rules:\n"
    "  - Think step-by-step in 'calc_chain' FIRST before writing anything else.\n"
    "  - CRITICAL: Inside 'calc_chain', enclose your final computed result in angle\n"
    "    brackets and END with it, e.g. <42>.\n"
    "  - Then copy that exact value into 'final_answer' (numeric only, no units).\n"
    "  - List chunk indices you actually used in 'evidence_ids'.\n"
    "Return JSON with keys IN THIS ORDER: calc_chain, evidence_ids, final_answer.\n"
    "\n"
    "CRITICAL MATH RULES:\n"
    "  - 'half as many' → divide by 2; 'twice as much' → multiply by 2.\n"
    "  - 'how much MORE does X need' = (what X needs) − (what X already has + gifts).\n"
    "  - 'X does Y to N people K times' → multiply by BOTH N and K.\n"
    "  - Count EVERY multiplier in the sentence — don't miss any."
)

FC_NOTOOL_SYSTEM_PROMPT_GEMINI = (
    "You are a careful math solver.\n"
    "Return STRICT JSON ONLY (no markdown, no prose outside JSON) with keys in this order:\n"
    "calc_chain, evidence_ids, final_answer.\n"
    "Requirements:\n"
    "  - calc_chain must be concise (<= 35 words) and end with <answer>.\n"
    "  - evidence_ids must list only used chunk indices.\n"
    "  - final_answer must copy exactly the value inside <answer>.\n"
    "Math rules:\n"
    "  - half as many => divide by 2; twice as much => multiply by 2.\n"
    "  - how much MORE => needed - already_have.\n"
    "  - multiply all explicit multipliers."
)

FC_NOTOOL_RESCUE_SYSTEM_PROMPT = (
    "Return STRICT JSON ONLY with keys in this exact order:\n"
    "calc_chain, evidence_ids, final_answer.\n"
    "Requirements:\n"
    "  - calc_chain must be ONE short sentence (max 25 words), end with <answer>.\n"
    "  - evidence_ids must be a list of integer chunk indices.\n"
    "  - final_answer must copy the value in <answer>.\n"
    "  - No markdown, no extra text."
)


def _is_gemini_backend(llm: LLMBackend) -> bool:
    """Detect Gemini via model name or Google OpenAI-compatible endpoint."""
    model_name = str(getattr(llm, "model", "") or "").lower()
    if "gemini" in model_name:
        return True
    try:
        base_url = str(getattr(getattr(llm, "client", None), "base_url", "") or "").lower()
    except Exception:
        base_url = ""
    return ("googleapis.com" in base_url) or ("generativelanguage" in base_url)


def _provider_kwargs(llm: LLMBackend) -> Dict[str, Any]:
    """Provider-specific request kwargs without changing existing call signatures."""
    extra: Dict[str, Any] = {}
    if "qwen" in llm.model.lower():
        extra["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
    # Gemini OpenAI-compatible API does not support seed.
    if not _is_gemini_backend(llm):
        extra["seed"] = llm.seed
    return extra


def _first_turn_tool_choice(llm: LLMBackend, force_tool: bool):
    """Keep old behavior, but use explicit function forcing for Gemini compatibility."""
    if not force_tool:
        return "auto"
    if _is_gemini_backend(llm):
        return {"type": "function", "function": {"name": "calculate"}}
    return "required"


def _tool_call_to_message_dict(tc: Any) -> Dict[str, Any]:
    """
    Convert SDK tool-call object to message dict while preserving provider extras.

    Gemini OpenAI-compat tool calls may include:
      extra_content.google.thought_signature
    which must be echoed back verbatim in subsequent turns.
    """
    # Best-effort: keep all fields (including unknown/provider-specific extras).
    try:
        if hasattr(tc, "model_dump"):
            d = tc.model_dump(exclude_none=True)
            if isinstance(d, dict):
                # Some SDK versions keep unknown fields in model_extra only.
                extra = getattr(tc, "model_extra", None)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        d.setdefault(k, v)
                return d
    except Exception:
        pass

    # Fallback to canonical OpenAI shape.
    return {
        "id": getattr(tc, "id", ""),
        "type": "function",
        "function": {
            "name": getattr(getattr(tc, "function", None), "name", ""),
            "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
        },
    }


def _chat_create_with_retry(
    llm: LLMBackend,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    max_attempts: int = 4,
):
    """Unified chat.completions call with lightweight retry for transient provider errors."""
    provider_kwargs = _provider_kwargs(llm)
    last_err = None
    for attempt in range(max_attempts):
        try:
            req: Dict[str, Any] = {
                "model": llm.model,
                "messages": messages,
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                **provider_kwargs,
            }
            if tools is not None:
                req["tools"] = tools
            if tool_choice is not None:
                req["tool_choice"] = tool_choice
            return llm.client.chat.completions.create(**req)
        except Exception as e:
            last_err = e
            if attempt >= max_attempts - 1:
                break
            msg = str(e).lower()
            retryable = any(k in msg for k in (
                "api key expired", "api_key_invalid",
                "rate limit", "429",
                "timeout", "timed out",
                "temporar", "unavailable",
                "500", "502", "503", "504",
            ))
            if not retryable:
                break
            time.sleep(min(4.0, 0.8 * (2 ** attempt)))
    raise RuntimeError(f"API call failed after {max_attempts} attempts: {last_err}")


def _extract_json_strict(text: str) -> Dict[str, Any]:
    """
    Parse JSON but disallow numeric fallback extraction.

    In notool modes, numeric fallback can accidentally pick chunk indices
    (e.g., 12/20/23) from non-JSON outputs, causing spurious answers.
    """
    try:
        out = extract_json(text)
    except Exception:
        return {}
    if not isinstance(out, dict):
        return {}
    if out.get("_parse_fallback"):
        return {}
    return out


def _extract_angle_answer(text: str, max_len: int = 60) -> str:
    """Extract answer from <42> or <answer>42</answer>, avoiding tag-name artifacts."""
    if not text:
        return ""
    xml_m = re.search(
        rf'(?is)<\s*(?:answer|result|final_answer)\s*>\s*([^<>]{{1,{max_len}}})\s*'
        rf'<\s*/\s*(?:answer|result|final_answer)\s*>',
        text,
    )
    if xml_m:
        return xml_m.group(1).strip()
    angle_matches = [m.strip() for m in re.findall(rf'<([^>]{{1,{max_len}}})>', text)]
    angle_matches = [
        m for m in angle_matches
        if not re.fullmatch(r'/?[A-Za-z_][\w:-]*', m)
    ]
    return angle_matches[-1] if angle_matches else ""


def _extract_scalar_field(raw_text: str, key: str, max_len: int = 2000) -> str:
    """Best-effort extraction for possibly malformed/truncated JSON scalar fields."""
    if not raw_text:
        return ""
    m = re.search(rf'(?is)["\']?{re.escape(key)}["\']?\s*[:=]\s*', raw_text)
    if not m:
        return ""
    tail = raw_text[m.end():].lstrip()
    if not tail:
        return ""

    if tail[0] in ('"', "'"):
        q = tail[0]
        body = tail[1:]
        close_idx = body.find(q)
        if close_idx != -1:
            val = body[:close_idx]
            return val.strip()[:max_len]
        cut = re.search(r"[\n,}]", body)
        if cut:
            return body[:cut.start()].strip()[:max_len]
        return body.strip()[:max_len]

    cut = re.search(r"[,}\n]", tail)
    val = tail[:cut.start()] if cut else tail
    return val.strip().strip('"').strip("'")[:max_len]


def _extract_int_list_field(raw_text: str, key: str, max_chars: int = 2000) -> List[int]:
    """Best-effort extraction for integer list fields like evidence_ids."""
    if not raw_text:
        return []
    m = re.search(rf'(?is)["\']?{re.escape(key)}["\']?\s*[:=]\s*\[', raw_text)
    if not m:
        return []
    tail = raw_text[m.end():]
    close_idx = tail.find("]")
    body = tail[:close_idx] if close_idx != -1 else tail[:max_chars]
    nums = re.findall(r"-?\d+", body)
    out: List[int] = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            continue
    return out


def _recover_fc_fields_from_raw(raw_text: str) -> Dict[str, Any]:
    """
    Recover final_answer/evidence_ids/reasoning from malformed model output.
    Used for Gemini-only fallback so non-Gemini behavior is unchanged.
    """
    reasoning = _extract_scalar_field(raw_text, "reasoning", max_len=4000)
    if not reasoning:
        reasoning = _extract_scalar_field(raw_text, "calc_chain", max_len=4000)
    final_answer = _extract_scalar_field(raw_text, "final_answer", max_len=120)
    if not final_answer:
        final_answer = _extract_angle_answer(raw_text, max_len=60)
    final_answer = re.sub(r"</?[^>]+>", "", final_answer).strip()
    return {
        "reasoning": reasoning,
        "evidence_ids": _extract_int_list_field(raw_text, "evidence_ids"),
        "final_answer": final_answer,
    }


DEFAULT_GSTEP_CONTINUE_PROMPT = (
    "Your previous tool result is: {prev_output}. "
    "Before finalizing, re-check the evidence chunks and verify:\n"
    "1) Did you use the correct numbers from evidence?\n"
    "2) Did you complete all required computation steps to the final answer?\n"
    "If anything is missing or incorrect, call calculate with a corrected and COMPLETE "
    "multistep expression (do not repeat the same expression). "
    "If already complete and consistent, return final JSON only."
)

DEFAULT_GSTEP_REPLAN_PROMPT = (
    "Re-check your calculation plan from evidence before finalizing. "
    "Your previous tool result is: {prev_output}. "
    "This may be stale or incorrect. "
    "If needed, call calculate with a revised full step decomposition "
    "(do not repeat the same step/expression). "
    "Then return final JSON only."
)

DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT = (
    "Your new calculation repeats a previous expression:\n"
    "{repeated_expression}\n"
    "That repeated expression previously returned: {prev_output}.\n"
    "Do NOT reuse the same expression. Re-read evidence and call calculate with a "
    "different, corrected, and COMPLETE multi-step expression. Then return final JSON only."
)

DEFAULT_GCOMMIT_ONE_MORE_PROMPT = (
    "Before finalizing, run one more verification pass. "
    "If your previous tool outputs are repetitive or uncertain, revise the expression and call calculate once. "
    "Do not repeat identical steps unless necessary. "
    "Then return final JSON only."
)

CRITIC_GSTEP_CONTINUE_PROMPT = (
    "Calculator returned: {prev_output}. "
    "Before calling calculate again, reason in words: "
    "which numbers from evidence do you need, and what is the correct step-by-step plan? "
    "Then call calculate with a CORRECTED expression (not the same one). "
    "Return final JSON after."
)

CRITIC_GSTEP_REPLAN_PROMPT = (
    "Previous result ({prev_output}) may be wrong. "
    "Re-derive the solution in words from evidence first, "
    "then call calculate with a revised expression. "
    "Do NOT reuse any previous expression. Return final JSON after."
)

CRITIC_GSTEP_DUPLICATE_REPLAN_PROMPT = (
    "You repeated: {repeated_expression} (returned {prev_output}). "
    "Explain in words why this is wrong and what should change, "
    "then call calculate with a DIFFERENT expression. Return final JSON after."
)

CRITIC_GCOMMIT_ONE_MORE_PROMPT = (
    "Verify in words: re-read the question, list key numbers from evidence, "
    "check each arithmetic step. Does the answer make sense? "
    "If wrong, call calculate with a corrected expression. Return final JSON after."
)


def _parse_fc_output_from_final_text(
    *,
    llm: LLMBackend,
    final_text: str,
    tool_trace: List[Dict[str, Any]],
    tool_mode: str,
    n_chunks_seen: int,
) -> Dict[str, Any]:
    """
    Parse the FC final text into standardized fields so the same logic can be
    reused after a commit-gate re-entry turn.
    """
    is_gemini_fc = _is_gemini_backend(llm)

    if is_gemini_fc:
        out = _extract_json_strict(final_text)
        if not out:
            out = {}
        recovered = _recover_fc_fields_from_raw(final_text)
        if (not isinstance(out.get("evidence_ids"), list)
                or not out.get("evidence_ids")):
            out["evidence_ids"] = recovered.get("evidence_ids", [])
        if not str(out.get("final_answer", "")).strip():
            out["final_answer"] = recovered.get("final_answer", "")
        if not str(out.get("reasoning", "")).strip():
            out["reasoning"] = recovered.get("reasoning", "") or final_text
    else:
        try:
            out = extract_json(final_text)
        except Exception as error:
            out = {
                "final_answer": "",
                "evidence_ids": [],
                "reasoning": "JSON parse error: {0}".format(error),
            }

    pred_evidence = out.get("evidence_ids", [])
    if not isinstance(pred_evidence, list):
        pred_evidence = []
    pred_evidence = [int(x) for x in pred_evidence if isinstance(x, (int, float))]

    pred_answer = str(out.get("final_answer", "")).strip()
    if is_gemini_fc and not pred_answer and out.get("reasoning"):
        pred_answer = _extract_angle_answer(str(out.get("reasoning", "")), max_len=30)
    if not pred_answer and tool_trace and tool_mode != "noop":
        last_calc = next(
            (t for t in reversed(tool_trace) if t.get("tool") == "calculator"),
            None,
        )
        if last_calc:
            out_str = str(last_calc.get("output", ""))
            if not out_str.startswith("Error") and not out_str.startswith("[CALCULATOR"):
                pred_answer = out_str

    expr_parts = []
    for tool_call in tool_trace:
        if tool_call.get("tool") == "calculator":
            calc_input = tool_call.get("input", "")
            if isinstance(calc_input, list):
                expr_parts.append(" ; ".join(str(step) for step in calc_input))
            else:
                expr_parts.append(str(calc_input))
    final_expr = " | ".join(expr_parts)

    return {
        "pred_answer": pred_answer,
        "pred_evidence_ids": pred_evidence,
        "reasoning": out.get("reasoning", ""),
        "expression": final_expr,
        "tool_trace": tool_trace,
        "n_chunks_seen": n_chunks_seen,
    }


# =========================================================================
# Core: FC loop with pluggable tool function
# =========================================================================

def _chat_with_tools_custom(
    llm: LLMBackend,
    system: str,
    user: str,
    tool_fn: Callable[[list], str],
    max_tool_calls: int = 5,
    force_tool: bool = True,
    g_step_gate: Optional[GStepGate] = None,
    g_step_max_extra_turns: int = 0,
    g_step_max_duplicate_replans: int = 1,
    g_step_continue_prompt: str = DEFAULT_GSTEP_CONTINUE_PROMPT,
    g_step_replan_prompt: str = DEFAULT_GSTEP_REPLAN_PROMPT,
    g_step_duplicate_replan_prompt: str = DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
    gate_force_tool_on_continue: bool = True,
    g_step_n_chunks_seen: int = 0,
    initial_messages: Optional[List[Dict[str, Any]]] = None,
    initial_tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """
    Real OpenAI function-calling loop with an injected tool function.

    Identical to LLMBackend.chat_with_tools() except the calculator logic
    is replaced by `tool_fn(calc_steps) -> str`, allowing controlled
    experiments (noop, perfect oracle, etc.) without touching agent.py.

    Returns
    -------
    (final_text, tool_trace, messages, gate_meta)
    """
    if initial_messages is not None:
        messages = copy(initial_messages)
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    tool_trace = copy(initial_tool_trace) if initial_tool_trace is not None else []
    last_choice = None
    g_step_continue_count = 0
    gstep_scores: List[float] = []
    g_step_no_progress_blocks = 0
    g_step_duplicate_blocks = 0
    g_step_duplicate_replan_used = 0
    g_step_block_reasons: Dict[str, int] = {
        "gate_predict_commit": 0,
        "max_extra_turns_reached": 0,
        "max_tool_calls_reached": 0,
        "no_progress_after_continue": 0,
        "gate_error": 0,
        "duplicate_expression_blocked": 0,
        "duplicate_expression_retry_exhausted": 0,
    }
    last_continue_marker = None
    force_tool_next_turn = False

    def _parse_llama_json_tool_calls(content: str):
        """
        Best-effort parse for Llama JSON tool-call outputs when vLLM parser fails.
        Expects JSON like {"name": "...", "arguments"/"parameters": {...}}
        or a list of such objects. Returns list of (name, args_dict, args_str).
        """
        if not content:
            return []
        # Strip Llama <|python_tag|> prefix if present
        if "<|python_tag|>" in content:
            content = content.split("<|python_tag|>", 1)[1].strip()
        try:
            obj = extract_json(content)
        except Exception:
            return []
        calls = obj if isinstance(obj, list) else [obj]
        parsed = []
        for c in calls:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("function")
            args = c.get("arguments", None)
            if args is None:
                args = c.get("parameters", None)
            if not name or args is None:
                continue
            # arguments may be a JSON string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if not isinstance(args, dict):
                continue
            parsed.append((name, args, json.dumps(args, ensure_ascii=False)))
        return parsed

    def _success_marker(trace):
        successful = []
        for tool_call in trace or []:
            output = str(tool_call.get("output", "")).strip()
            if output and (not output.startswith("Error")) and (not output.startswith("[CALCULATOR")):
                successful.append(output)
        return (len(successful), successful[-1] if successful else "")

    def _stagnating_outputs(trace) -> bool:
        successful = []
        for tool_call in trace or []:
            output = str(tool_call.get("output", "")).strip()
            if output and (not output.startswith("Error")) and (not output.startswith("[CALCULATOR")):
                successful.append(output)
        if len(successful) < 2:
            return False
        return (successful[-1] == successful[-2]) or (len(set(successful)) == 1)

    def _calc_steps_to_text(calc_steps) -> str:
        if isinstance(calc_steps, list):
            return " ; ".join(str(step) for step in calc_steps)
        return str(calc_steps)

    def _normalize_expression(expr_text: str) -> str:
        expr = str(expr_text or "").strip().lower()
        expr = re.sub(r"\s+", "", expr)
        return expr

    def _expression_history(trace):
        history: Dict[str, Dict[str, str]] = {}
        for tool_call in trace or []:
            if tool_call.get("tool") != "calculator":
                continue
            raw_expr = _calc_steps_to_text(tool_call.get("input", []))
            signature = _normalize_expression(raw_expr)
            if not signature:
                continue
            history[signature] = {
                "raw": raw_expr,
                "output": str(tool_call.get("output", "")).strip(),
            }
        return history

    def _last_success_output(trace) -> str:
        for tool_call in reversed(trace or []):
            output = str(tool_call.get("output", "")).strip()
            if output and (not output.startswith("Error")) and (not output.startswith("[CALCULATOR")):
                return output
        return "N/A"

    def _format_continue_prompt(template: str, trace, turn_idx: int) -> str:
        values = {
            "prev_output": _last_success_output(trace),
            "turn": int(turn_idx),
            "max_tool_calls": int(max_tool_calls),
            "n_chunks_seen": int(g_step_n_chunks_seen),
        }
        try:
            return template.format(**values)
        except Exception:
            return template

    def _format_duplicate_replan_prompt(
        template: str,
        repeated_expression: str,
        prev_output: str,
    ) -> str:
        values = {
            "repeated_expression": repeated_expression or "N/A",
            "prev_output": prev_output or "N/A",
        }
        try:
            return template.format(**values)
        except Exception:
            return template

    for turn in range(max_tool_calls + 1):
        if force_tool_next_turn:
            tool_choice = _first_turn_tool_choice(llm, True)
            force_tool_next_turn = False
        elif turn == 0:
            tool_choice = _first_turn_tool_choice(llm, force_tool)
        else:
            tool_choice = "auto"

        try:
            resp = _chat_create_with_retry(
                llm=llm,
                messages=messages,
                tools=[_CALCULATE_TOOL_SCHEMA],
                tool_choice=tool_choice,
            )
        except Exception as e:
            raise RuntimeError(f"FC API call failed at turn {turn}: {e}")

        last_choice = resp.choices[0]

        if last_choice.message.tool_calls:
            # ---- Execute tool calls ----
            tool_calls = last_choice.message.tool_calls
            if g_step_gate is not None and tool_calls:
                history = _expression_history(tool_trace)
                repeated_expr = ""
                previous_output = ""
                for tool_call in tool_calls:
                    if tool_call.function.name != "calculate":
                        continue
                    try:
                        args_tmp = json.loads(tool_call.function.arguments)
                        calc_steps_tmp = args_tmp.get("calc_steps", [])
                    except Exception:
                        calc_steps_tmp = []
                    raw_expr_tmp = _calc_steps_to_text(calc_steps_tmp)
                    signature_tmp = _normalize_expression(raw_expr_tmp)
                    if signature_tmp and signature_tmp in history:
                        repeated_expr = raw_expr_tmp or history[signature_tmp].get("raw", "")
                        previous_output = history[signature_tmp].get("output", "")
                        break

                if repeated_expr:
                    if (
                        g_step_duplicate_replan_used < int(max(0, g_step_max_duplicate_replans))
                        and turn < max_tool_calls
                    ):
                        g_step_duplicate_blocks += 1
                        g_step_duplicate_replan_used += 1
                        g_step_block_reasons["duplicate_expression_blocked"] += 1
                        prompt_text = _format_duplicate_replan_prompt(
                            g_step_duplicate_replan_prompt or DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
                            repeated_expr,
                            previous_output,
                        )
                        messages.append({
                            "role": "assistant",
                            "content": last_choice.message.content,
                        })
                        messages.append({
                            "role": "user",
                            "content": prompt_text,
                        })
                        continue
                    g_step_block_reasons["duplicate_expression_retry_exhausted"] += 1

            if _is_gemini_backend(llm):
                tool_calls_dict = [_tool_call_to_message_dict(tc) for tc in tool_calls]
            else:
                tool_calls_dict = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append({
                "role": "assistant",
                "content": last_choice.message.content,
                "tool_calls": tool_calls_dict,
            })

            for tc in tool_calls:
                if tc.function.name == "calculate":
                    try:
                        args = json.loads(tc.function.arguments)
                        calc_steps = args.get("calc_steps", [])
                        result_str = tool_fn(calc_steps)
                    except Exception as e:
                        calc_steps = []
                        result_str = f"Error: {e}"

                    tool_trace.append({
                        "tool": "calculator",
                        "input": calc_steps,
                        "output": result_str,
                        "tool_call_id": tc.id,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: unknown tool '{tc.function.name}'",
                    })

        else:
            content = last_choice.message.content or ""
            # Fallback: Qwen3 vLLM XML tool-call bug
            xml_calls = _parse_xml_tool_calls(content)
            if xml_calls:
                if g_step_gate is not None:
                    history = _expression_history(tool_trace)
                    repeated_expr = ""
                    previous_output = ""
                    for name_tmp, args_tmp, _ in xml_calls:
                        if name_tmp != "calculate":
                            continue
                        calc_steps_tmp = args_tmp.get("calc_steps", []) if isinstance(args_tmp, dict) else []
                        raw_expr_tmp = _calc_steps_to_text(calc_steps_tmp)
                        signature_tmp = _normalize_expression(raw_expr_tmp)
                        if signature_tmp and signature_tmp in history:
                            repeated_expr = raw_expr_tmp or history[signature_tmp].get("raw", "")
                            previous_output = history[signature_tmp].get("output", "")
                            break
                    if repeated_expr:
                        if (
                            g_step_duplicate_replan_used < int(max(0, g_step_max_duplicate_replans))
                            and turn < max_tool_calls
                        ):
                            g_step_duplicate_blocks += 1
                            g_step_duplicate_replan_used += 1
                            g_step_block_reasons["duplicate_expression_blocked"] += 1
                            prompt_text = _format_duplicate_replan_prompt(
                                g_step_duplicate_replan_prompt or DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
                                repeated_expr,
                                previous_output,
                            )
                            messages.append({
                                "role": "assistant",
                                "content": content,
                            })
                            messages.append({
                                "role": "user",
                                "content": prompt_text,
                            })
                            continue
                        g_step_block_reasons["duplicate_expression_retry_exhausted"] += 1
                tool_calls_dict = []
                for i, (name, args, _) in enumerate(xml_calls):
                    tool_calls_dict.append({
                        "id": f"fallback_xml_{turn}_{i}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    })
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls_dict,
                })
                for i, (name, args, _) in enumerate(xml_calls):
                    tc_id = tool_calls_dict[i]["id"]
                    if name == "calculate":
                        try:
                            calc_steps = args.get("calc_steps", [])
                            result_str = tool_fn(calc_steps)
                        except Exception as e:
                            calc_steps = []
                            result_str = f"Error: {e}"
                        tool_trace.append({
                            "tool": "calculator",
                            "input": calc_steps,
                            "output": result_str,
                            "tool_call_id": tc_id,
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result_str,
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": f"Error: unknown tool '{name}'",
                        })
            else:
                # Fallback: Llama JSON tool-call output (when vLLM parser fails)
                llama_calls = _parse_llama_json_tool_calls(content)
                if llama_calls:
                    if g_step_gate is not None:
                        history = _expression_history(tool_trace)
                        repeated_expr = ""
                        previous_output = ""
                        for name_tmp, args_tmp, _ in llama_calls:
                            if name_tmp != "calculate":
                                continue
                            calc_steps_tmp = args_tmp.get("calc_steps", []) if isinstance(args_tmp, dict) else []
                            raw_expr_tmp = _calc_steps_to_text(calc_steps_tmp)
                            signature_tmp = _normalize_expression(raw_expr_tmp)
                            if signature_tmp and signature_tmp in history:
                                repeated_expr = raw_expr_tmp or history[signature_tmp].get("raw", "")
                                previous_output = history[signature_tmp].get("output", "")
                                break
                        if repeated_expr:
                            if (
                                g_step_duplicate_replan_used < int(max(0, g_step_max_duplicate_replans))
                                and turn < max_tool_calls
                            ):
                                g_step_duplicate_blocks += 1
                                g_step_duplicate_replan_used += 1
                                g_step_block_reasons["duplicate_expression_blocked"] += 1
                                prompt_text = _format_duplicate_replan_prompt(
                                    g_step_duplicate_replan_prompt or DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
                                    repeated_expr,
                                    previous_output,
                                )
                                messages.append({
                                    "role": "assistant",
                                    "content": content,
                                })
                                messages.append({
                                    "role": "user",
                                    "content": prompt_text,
                                })
                                continue
                            g_step_block_reasons["duplicate_expression_retry_exhausted"] += 1
                    tool_calls_dict = []
                    for i, (name, args, _) in enumerate(llama_calls):
                        tool_calls_dict.append({
                            "id": f"fallback_llama_{turn}_{i}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        })
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls_dict,
                    })
                    for i, (name, args, _) in enumerate(llama_calls):
                        tc_id = tool_calls_dict[i]["id"]
                        if name == "calculate":
                            try:
                                calc_steps = args.get("calc_steps", [])
                                result_str = tool_fn(calc_steps)
                            except Exception as e:
                                calc_steps = []
                                result_str = f"Error: {e}"
                            tool_trace.append({
                                "tool": "calculator",
                                "input": calc_steps,
                                "output": result_str,
                                "tool_call_id": tc_id,
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result_str,
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": f"Error: unknown tool '{name}'",
                            })
                else:
                    # Candidate final response — no more tool calls
                    final_text = strip_think_tags(content).strip()
                    if _is_gemini_backend(llm) and (not final_text):
                        fr = getattr(last_choice, "finish_reason", "")
                        raise RuntimeError(
                            f"Empty Gemini response at turn {turn} (finish_reason={fr})"
                        )

                    no_progress_after_continue = False
                    if last_continue_marker is not None:
                        current_marker = _success_marker(tool_trace)
                        if current_marker == last_continue_marker:
                            no_progress_after_continue = True
                        else:
                            last_continue_marker = None

                    if g_step_gate is not None:
                        if no_progress_after_continue:
                            g_step_no_progress_blocks += 1
                            g_step_block_reasons["no_progress_after_continue"] += 1
                        elif g_step_continue_count >= int(max(0, g_step_max_extra_turns)):
                            g_step_block_reasons["max_extra_turns_reached"] += 1
                        elif turn >= max_tool_calls:
                            g_step_block_reasons["max_tool_calls_reached"] += 1
                        else:
                            try:
                                step_decision = g_step_gate.decide(
                                    tool_trace=tool_trace,
                                    candidate_text=final_text,
                                    n_chunks_seen=g_step_n_chunks_seen,
                                    max_tool_calls_limit=max_tool_calls,
                                    extra_turns_used=g_step_continue_count,
                                )
                                gstep_scores.append(float(step_decision.p_continue))
                                if step_decision.action == "continue":
                                    prompt_text = g_step_continue_prompt or DEFAULT_GSTEP_CONTINUE_PROMPT
                                    if _stagnating_outputs(tool_trace):
                                        prompt_text = g_step_replan_prompt or DEFAULT_GSTEP_REPLAN_PROMPT
                                    prompt_text = _format_continue_prompt(prompt_text, tool_trace, turn)
                                    last_continue_marker = _success_marker(tool_trace)
                                    messages.append({"role": "assistant", "content": content})
                                    messages.append({
                                        "role": "user",
                                        "content": prompt_text,
                                    })
                                    g_step_continue_count += 1
                                    force_tool_next_turn = bool(gate_force_tool_on_continue)
                                    continue
                                g_step_block_reasons["gate_predict_commit"] += 1
                            except Exception:
                                g_step_block_reasons["gate_error"] += 1

                    messages.append({"role": "assistant", "content": content})
                    return final_text, tool_trace, messages, {
                        "g_step_continue_count": g_step_continue_count,
                        "g_step_no_progress_blocks": g_step_no_progress_blocks,
                        "g_step_duplicate_blocks": g_step_duplicate_blocks,
                        "g_step_block_reasons": g_step_block_reasons,
                        "g_step_scores": gstep_scores,
                    }

    # Exhausted max_tool_calls
    final_text = (last_choice.message.content or "") if last_choice else ""
    final_text = strip_think_tags(final_text).strip()
    if final_text:
        messages.append({"role": "assistant", "content": final_text})
    return final_text, tool_trace, messages, {
        "g_step_continue_count": g_step_continue_count,
        "g_step_no_progress_blocks": g_step_no_progress_blocks,
        "g_step_duplicate_blocks": g_step_duplicate_blocks,
        "g_step_block_reasons": g_step_block_reasons,
        "g_step_scores": gstep_scores,
    }


# =========================================================================
# Agent with configurable tool behavior
# =========================================================================

class MechanismFCAgent:
    """
    FC agent for mechanism ablation.

    Parameters
    ----------
    tool_mode : str
        'real'    — standard calculator (safe_calculate_steps)
        'noop'    — echoes expression back without computing
        'perfect' — always returns the gold answer (injected at solve time)

    max_tool_calls : int
        Maximum number of tool call iterations. Set to 1 to test H2.

    oracle_evidence : bool
        If True, present only ground-truth evidence chunks (no noise chunks).
        Tests H1: how much does noise hurt evidence selection?
    """

    def __init__(
        self,
        llm: LLMBackend,
        tool_mode: str = "real",
        max_tool_calls: int = 5,
        oracle_evidence: bool = False,
        force_tool_first_turn: bool = True,
        g_step_gate: Optional[GStepGate] = None,
        g_commit_gate: Optional[GCommitGate] = None,
        g_step_max_extra_turns: int = 3,
        g_step_continue_prompt: str = DEFAULT_GSTEP_CONTINUE_PROMPT,
        g_step_replan_prompt: str = DEFAULT_GSTEP_REPLAN_PROMPT,
        g_step_duplicate_replan_prompt: str = DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
        g_commit_one_more_prompt: str = DEFAULT_GCOMMIT_ONE_MORE_PROMPT,
        gate_force_tool_on_continue: bool = True,
    ):
        assert tool_mode in ("real", "noop", "perfect", "notool", "fcprompt"), \
            f"tool_mode must be 'real','noop','perfect','notool','fcprompt'; got '{tool_mode}'"
        self.llm = llm
        self.tool_mode = tool_mode
        self.max_tool_calls = max_tool_calls
        self.oracle_evidence = oracle_evidence
        self.force_tool_first_turn = force_tool_first_turn
        self.g_step_gate = g_step_gate
        self.g_commit_gate = g_commit_gate
        self.g_step_max_extra_turns = max(0, int(g_step_max_extra_turns))
        self.g_step_continue_prompt = g_step_continue_prompt
        self.g_step_replan_prompt = g_step_replan_prompt
        self.g_step_duplicate_replan_prompt = g_step_duplicate_replan_prompt
        self.g_commit_one_more_prompt = g_commit_one_more_prompt
        self.gate_force_tool_on_continue = bool(gate_force_tool_on_continue)

    @staticmethod
    def _mock_calc_steps(calc_chain: Optional[str]) -> List[str]:
        if not calc_chain:
            return []
        steps = []
        for part in calc_chain.split(";"):
            candidate = part.strip()
            if not candidate or candidate.startswith("final="):
                continue
            steps.append(candidate)
        return steps

    def _solve_mock(
        self,
        env: SingleTurnEnvironment,
        result_chunks: List[Any],
        gold_answer: str,
    ) -> Dict[str, Any]:
        calc_steps = self._mock_calc_steps(env.problem.calc_chain)
        tool_trace = []
        if self.tool_mode in {"real", "perfect", "noop"}:
            if self.tool_mode == "noop":
                tool_output = "[CALCULATOR OFFLINE] Mock backend skipped external computation."
            elif self.tool_mode == "perfect":
                tool_output = str(gold_answer)
            else:
                tool_output = safe_calculate_steps(calc_steps) if calc_steps else str(gold_answer)
            tool_trace.append(
                {
                    "tool": "calculator",
                    "input": calc_steps,
                    "output": tool_output,
                    "tool_call_id": "mock_tool_call_0",
                }
            )

        reasoning = (
            "Mock backend used the gold evidence and a deterministic answer path "
            "for smoke-test verification."
        )
        gate_action = "submit"
        gate_p_continue = 0.0
        gate_p_one_more = 0.0
        gate_anchor_applied = False
        gate_one_more_used = 0
        gate_step_continue_count = 0
        gate_step_no_progress_blocks = 0
        gate_step_duplicate_blocks = 0
        gate_step_block_reasons: Dict[str, int] = {}

        if self.tool_mode == "real" and self.g_step_gate is not None:
            try:
                step_decision = self.g_step_gate.decide(
                    tool_trace=tool_trace,
                    candidate_text=reasoning,
                    pred_answer=str(gold_answer),
                    n_chunks_seen=len(result_chunks),
                    max_tool_calls_limit=self.max_tool_calls,
                    extra_turns_used=0,
                )
                gate_p_continue = float(step_decision.p_continue)
                gate_step_continue_count = 1 if step_decision.action == "continue" else 0
            except Exception:
                pass

        if self.tool_mode == "real" and self.g_commit_gate is not None:
            try:
                commit_decision = self.g_commit_gate.decide(
                    pred_answer=str(gold_answer),
                    reasoning=reasoning,
                    tool_trace=tool_trace,
                    n_chunks_seen=len(result_chunks),
                    pred_evidence_ids=list(env.problem.evidence_ids),
                    expression=" ; ".join(calc_steps),
                    max_tool_calls_limit=self.max_tool_calls,
                    one_more_used=0,
                )
                gate_action = commit_decision.action
                gate_p_one_more = float(commit_decision.p_one_more)
                gate_anchor_applied = bool(commit_decision.anchor_applied)
                gate_one_more_used = 1 if commit_decision.action == "one_more_turn" else 0
            except Exception:
                pass

        return {
            "pred_answer": str(gold_answer),
            "pred_evidence_ids": list(env.problem.evidence_ids),
            "reasoning": reasoning,
            "expression": " ; ".join(calc_steps),
            "tool_trace": tool_trace,
            "n_chunks_seen": len(result_chunks),
            "env_stats": env.get_stats(),
            "gate_action": gate_action,
            "gate_p_continue": gate_p_continue,
            "gate_p_one_more": gate_p_one_more,
            "gate_anchor_applied": gate_anchor_applied,
            "gate_one_more_used": gate_one_more_used,
            "gate_step_continue_count": gate_step_continue_count,
            "gate_step_no_progress_blocks": gate_step_no_progress_blocks,
            "gate_step_duplicate_blocks": gate_step_duplicate_blocks,
            "gate_step_block_reasons": gate_step_block_reasons,
        }

    def solve(self, env: SingleTurnEnvironment) -> Dict[str, Any]:
        question = env.get_question()
        gold = env.get_gold()
        gold_answer = gold["answer"]

        # ---- Evidence selection: noisy (default) or oracle (evidence only) ----
        if self.oracle_evidence:
            # Only ground-truth evidence chunks, original indices preserved
            result_chunks = list(env.problem.evidence_chunks)
        else:
            result = env.retrieve_all()
            result_chunks = result.chunks

        chunk_text = _numbered_block(result_chunks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Information chunks:\n{chunk_text}\n"
        )

        if getattr(self.llm, "is_mock", False):
            return self._solve_mock(env, result_chunks, gold_answer)

        # ---- No-tool CoT mode: skip FC loop entirely ----
        if self.tool_mode == "notool":
            import re as _re
            is_gemini_notool = _is_gemini_backend(self.llm)

            def _extract_bracket_answer(text: str, max_len: int = 60) -> str:
                """Extract answer from <42> or <answer>42</answer>, avoiding tag-name artifacts."""
                if not text:
                    return ""
                xml_m = _re.search(
                    rf'(?is)<\s*(?:answer|result|final_answer)\s*>\s*([^<>]{{1,{max_len}}})\s*'
                    rf'<\s*/\s*(?:answer|result|final_answer)\s*>',
                    text,
                )
                if xml_m:
                    return xml_m.group(1).strip()
                angle_matches = [m.strip() for m in _re.findall(rf'<([^>]{{1,{max_len}}})>', text)]
                # Drop pure tag names like "answer" or "/answer".
                angle_matches = [
                    m for m in angle_matches
                    if not _re.fullmatch(r'/?[A-Za-z_][\w:-]*', m)
                ]
                return angle_matches[-1] if angle_matches else ""

            # User message: show chunks THEN question, JSON template with calc_chain first
            notool_user = (
                f"Information chunks:\n{chunk_text}\n\n"
                f"Question: {question}\n\n"
                "Respond with JSON (KEEP THIS FIELD ORDER):\n"
                "{\n"
                '  "calc_chain": "step-by-step reasoning ending with <result>",\n'
                '  "evidence_ids": [integer chunk indices used],\n'
                '  "final_answer": "numeric value copied from <result> in calc_chain"\n'
                "}"
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        FC_NOTOOL_SYSTEM_PROMPT_GEMINI
                        if is_gemini_notool else FC_NOTOOL_SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": notool_user},
            ]
            try:
                resp = _chat_create_with_retry(
                    llm=self.llm,
                    messages=messages,
                )
                final_text = strip_think_tags(
                    resp.choices[0].message.content or ""
                ).strip()
            except Exception as e:
                return {
                    "pred_answer": "",
                    "pred_evidence_ids": [],
                    "reasoning": f"ERROR: {e}",
                    "expression": "",
                    "tool_trace": [],
                    "n_chunks_seen": len(result_chunks),
                    "env_stats": env.get_stats(),
                }
            raw_notool_text = final_text
            if is_gemini_notool:
                out = _extract_json_strict(final_text)
            else:
                try:
                    out = extract_json(final_text)
                except Exception:
                    out = {}

            if (is_gemini_notool and (
                    not out
                    or "calc_chain" not in out
                    or "final_answer" not in out)):
                repair_messages = messages + [
                    {"role": "assistant", "content": final_text},
                    {"role": "user", "content": (
                        "Your previous response was not valid JSON.\n"
                        "Return STRICT JSON ONLY with keys in this exact order:\n"
                        "calc_chain, evidence_ids, final_answer.\n"
                        "Do not output any extra text."
                    )},
                ]
                try:
                    repair_resp = _chat_create_with_retry(
                        llm=self.llm,
                        messages=repair_messages,
                    )
                    repaired_text = strip_think_tags(
                        repair_resp.choices[0].message.content or ""
                    ).strip()
                    repaired_out = _extract_json_strict(repaired_text)
                    if repaired_out:
                        final_text = repaired_text
                        raw_notool_text = repaired_text
                        out = repaired_out
                    elif repaired_text:
                        raw_notool_text = repaired_text
                except Exception:
                    pass
            pred_evidence = out.get("evidence_ids", [])
            if not isinstance(pred_evidence, list):
                pred_evidence = []
            pred_evidence = [int(x) for x in pred_evidence
                             if isinstance(x, (int, float))]
            # Conservative fallback: parse explicit evidence_ids list if present in raw text.
            if is_gemini_notool and not pred_evidence and raw_notool_text:
                em = _re.search(
                    r'(?is)["\']?evidence_ids["\']?\s*[:=]\s*\[([^\]]*)\]',
                    raw_notool_text,
                )
                if em:
                    pred_evidence = [int(x) for x in _re.findall(r'-?\d+', em.group(1))]
            pred_answer = str(out.get("final_answer", "")).strip()
            calc_chain = out.get("calc_chain", "")
            if not calc_chain:
                calc_chain = raw_notool_text
            # Fallback 1: extract from last <...> in calc_chain if final_answer is empty
            if not pred_answer and calc_chain:
                pred_answer = _extract_bracket_answer(calc_chain, max_len=30)
            # Fallback 1.5: parse explicit final_answer field from raw non-JSON text.
            if is_gemini_notool and not pred_answer and raw_notool_text:
                fm = _re.search(
                    r'(?is)["\']?final[_\s]?answer["\']?\s*[:=]\s*["\']?([^"\',}\n]+)',
                    raw_notool_text,
                )
                if fm:
                    pred_answer = fm.group(1).strip()
            # Final rescue: one concise retry to avoid truncation-driven empty answers.
            if is_gemini_notool and (not pred_answer or not pred_evidence):
                rescue_user = (
                    f"Question: {question}\n\n"
                    f"Information chunks:\n{chunk_text}\n"
                )
                try:
                    rescue_resp = _chat_create_with_retry(
                        llm=self.llm,
                        messages=[
                            {"role": "system", "content": FC_NOTOOL_RESCUE_SYSTEM_PROMPT},
                            {"role": "user", "content": rescue_user},
                        ],
                    )
                    rescue_text = strip_think_tags(
                        rescue_resp.choices[0].message.content or ""
                    ).strip()
                    rescue_out = _extract_json_strict(rescue_text)
                    if rescue_out:
                        raw_notool_text = rescue_text
                        calc_chain = str(rescue_out.get("calc_chain", "")).strip() or calc_chain
                        ans2 = str(rescue_out.get("final_answer", "")).strip()
                        if ans2:
                            pred_answer = ans2
                        ev2 = rescue_out.get("evidence_ids", [])
                        if isinstance(ev2, list):
                            ev2 = [int(x) for x in ev2 if isinstance(x, (int, float))]
                            if ev2:
                                pred_evidence = ev2
                    if not pred_answer and rescue_text:
                        # Last attempt: parse <answer> from rescue raw text.
                        pred_answer = _extract_bracket_answer(rescue_text, max_len=30)
                except Exception:
                    pass
            # Fallback 2 (non-Gemini only): if final_answer is non-empty but does NOT
            # appear in calc_chain, some models may have copied incorrectly. In that
            # case, trust calc_chain's last number.
            #
            # Gemini responses often mention chunk indices in calc_chain; using last
            # number fallback can incorrectly pick those indices as answers.
            if (not is_gemini_notool) and pred_answer and calc_chain and pred_answer not in calc_chain:
                nums_in_chain = _re.findall(r'\b(\d+(?:\.\d+)?)\b', calc_chain)
                if nums_in_chain:
                    last_num = nums_in_chain[-1]
                    try:
                        val = float(last_num)
                        if val == int(val):
                            last_num = str(int(val))
                    except ValueError:
                        pass
                    pred_answer = last_num
            return {
                "pred_answer": pred_answer,
                "pred_evidence_ids": pred_evidence,
                "reasoning": calc_chain or out.get("reasoning", "") or raw_notool_text,
                "expression": "",
                "tool_trace": [],
                "n_chunks_seen": len(result_chunks),
                "env_stats": env.get_stats(),
            }

        # ---- FC-prompt, no tools: same FC_SYSTEM_PROMPT as baseline but no tools in API ----
        # Purpose: isolates prompt-style effect from tool-availability effect.
        # cf. fc_noop (FC prompt + forced tool call + offline) vs
        #     fc_notool_cot (CoT prompt + no tools)
        if self.tool_mode == "fcprompt":
            messages = [
                {"role": "system", "content": FC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                resp = _chat_create_with_retry(
                    llm=self.llm,
                    messages=messages,
                )
                final_text = strip_think_tags(
                    resp.choices[0].message.content or ""
                ).strip()
            except Exception as e:
                return {
                    "pred_answer": "",
                    "pred_evidence_ids": [],
                    "reasoning": f"ERROR: {e}",
                    "expression": "",
                    "tool_trace": [],
                    "n_chunks_seen": len(result_chunks),
                    "env_stats": env.get_stats(),
                }
            is_gemini_fc = _is_gemini_backend(self.llm)
            if is_gemini_fc:
                out = _extract_json_strict(final_text)
                if not out:
                    out = {}
                recovered = _recover_fc_fields_from_raw(final_text)
                if (not isinstance(out.get("evidence_ids"), list)
                        or not out.get("evidence_ids")):
                    out["evidence_ids"] = recovered.get("evidence_ids", [])
                if not str(out.get("final_answer", "")).strip():
                    out["final_answer"] = recovered.get("final_answer", "")
                if not str(out.get("reasoning", "")).strip():
                    out["reasoning"] = recovered.get("reasoning", "") or final_text
            else:
                try:
                    out = extract_json(final_text)
                except Exception:
                    out = {}
            pred_evidence = out.get("evidence_ids", [])
            if not isinstance(pred_evidence, list):
                pred_evidence = []
            pred_evidence = [int(x) for x in pred_evidence
                             if isinstance(x, (int, float))]
            pred_answer = str(out.get("final_answer", "")).strip()
            reasoning  = out.get("reasoning", "")
            if is_gemini_fc and not pred_answer and reasoning:
                pred_answer = _extract_angle_answer(reasoning, max_len=30)
            # Non-Gemini fallback: if final_answer empty, try last number in reasoning.
            # Gemini keeps this disabled to avoid chunk-index contamination.
            if (not is_gemini_fc) and (not pred_answer) and reasoning:
                import re as _re
                nums = _re.findall(r'-?\d+(?:\.\d+)?', reasoning)
                if nums:
                    pred_answer = nums[-1]
            return {
                "pred_answer": pred_answer,
                "pred_evidence_ids": pred_evidence,
                "reasoning": reasoning,
                "expression": "",
                "tool_trace": [],
                "n_chunks_seen": len(result_chunks),
                "env_stats": env.get_stats(),
            }

        # ---- Tool function based on mode ----
        if self.tool_mode == "noop":
            def tool_fn(calc_steps: list) -> str:
                expr = " ; ".join(str(s) for s in calc_steps)
                return (
                    f"[CALCULATOR OFFLINE] Expression received: {expr}. "
                    f"The calculator is unavailable — you must compute the "
                    f"final numeric value mentally and include it in your JSON answer."
                )
        elif self.tool_mode == "perfect":
            def tool_fn(calc_steps: list) -> str:
                # Always return the gold answer regardless of expression
                return str(gold_answer)
        else:
            calc_state: Dict[str, Any] = {}

            def tool_fn(calc_steps: list) -> str:
                # Persist calculator variables across FC turns for this sample.
                return safe_calculate_steps(calc_steps, state=calc_state)

        # ---- Run FC loop ----
        try:
            final_text, tool_trace, session_messages, step_meta = _chat_with_tools_custom(
                llm=self.llm,
                system=FC_SYSTEM_PROMPT,
                user=user_prompt,
                tool_fn=tool_fn,
                max_tool_calls=self.max_tool_calls,
                force_tool=self.force_tool_first_turn,
                g_step_gate=self.g_step_gate if self.tool_mode == "real" else None,
                g_step_max_extra_turns=self.g_step_max_extra_turns,
                g_step_continue_prompt=self.g_step_continue_prompt,
                g_step_replan_prompt=self.g_step_replan_prompt,
                g_step_duplicate_replan_prompt=self.g_step_duplicate_replan_prompt,
                gate_force_tool_on_continue=self.gate_force_tool_on_continue,
                g_step_n_chunks_seen=len(result_chunks),
            )
        except Exception as e:
            return {
                "pred_answer": "",
                "pred_evidence_ids": [],
                "reasoning": f"ERROR: {e}",
                "expression": "",
                "tool_trace": [],
                "n_chunks_seen": len(result_chunks),
                "env_stats": env.get_stats(),
            }

        parsed = _parse_fc_output_from_final_text(
            llm=self.llm,
            final_text=final_text,
            tool_trace=tool_trace,
            tool_mode=self.tool_mode,
            n_chunks_seen=len(result_chunks),
        )
        pred_answer = parsed["pred_answer"]
        pred_evidence = parsed["pred_evidence_ids"]
        reasoning = parsed["reasoning"]
        final_expr = parsed["expression"]

        gate_action = "submit"
        gate_p_continue = 0.0
        gate_p_one_more = 0.0
        gate_anchor_applied = False
        one_more_used = 0
        if step_meta and step_meta.get("g_step_scores"):
            gate_p_continue = float(step_meta["g_step_scores"][-1])

        if self.tool_mode == "real" and self.g_commit_gate is not None:
            try:
                commit_decision = self.g_commit_gate.decide(
                    pred_answer=pred_answer,
                    reasoning=reasoning,
                    tool_trace=tool_trace,
                    n_chunks_seen=len(result_chunks),
                    pred_evidence_ids=pred_evidence,
                    expression=final_expr,
                    max_tool_calls_limit=self.max_tool_calls,
                    one_more_used=0,
                )
                gate_action = commit_decision.action
                gate_p_one_more = float(commit_decision.p_one_more)
                gate_anchor_applied = bool(commit_decision.anchor_applied)
                pred_answer = commit_decision.pred_answer

                if commit_decision.action == "one_more_turn":
                    one_more_used = 1
                    follow_messages = copy(session_messages)
                    follow_messages.append({
                        "role": "user",
                        "content": self.g_commit_one_more_prompt or DEFAULT_GCOMMIT_ONE_MORE_PROMPT,
                    })
                    final_text2, tool_trace2, _messages2, _step_meta2 = _chat_with_tools_custom(
                        llm=self.llm,
                        system=FC_SYSTEM_PROMPT,
                        user=user_prompt,
                        tool_fn=tool_fn,
                        max_tool_calls=1,
                        force_tool=True,
                        g_step_gate=None,
                        g_step_n_chunks_seen=len(result_chunks),
                        initial_messages=follow_messages,
                        initial_tool_trace=tool_trace,
                    )
                    parsed2 = _parse_fc_output_from_final_text(
                        llm=self.llm,
                        final_text=final_text2,
                        tool_trace=tool_trace2,
                        tool_mode=self.tool_mode,
                        n_chunks_seen=len(result_chunks),
                    )
                    pred_answer = parsed2["pred_answer"]
                    pred_evidence = parsed2["pred_evidence_ids"]
                    reasoning = parsed2["reasoning"]
                    final_expr = parsed2["expression"]
                    tool_trace = tool_trace2
            except Exception:
                pass

        return {
            "pred_answer": pred_answer,
            "pred_evidence_ids": pred_evidence,
            "reasoning": reasoning,
            "expression": final_expr,
            "tool_trace": tool_trace,
            "n_chunks_seen": len(result_chunks),
            "env_stats": env.get_stats(),
            "gate_action": gate_action,
            "gate_p_continue": gate_p_continue,
            "gate_p_one_more": gate_p_one_more,
            "gate_anchor_applied": gate_anchor_applied,
            "gate_one_more_used": one_more_used,
            "gate_step_continue_count": int(step_meta.get("g_step_continue_count", 0)) if isinstance(step_meta, dict) else 0,
            "gate_step_no_progress_blocks": int(step_meta.get("g_step_no_progress_blocks", 0)) if isinstance(step_meta, dict) else 0,
            "gate_step_duplicate_blocks": int(step_meta.get("g_step_duplicate_blocks", 0)) if isinstance(step_meta, dict) else 0,
            "gate_step_block_reasons": step_meta.get("g_step_block_reasons", {}) if isinstance(step_meta, dict) else {},
        }


# =========================================================================
# Evaluation helper
# =========================================================================

def _eval_mechanism(prob, agent: MechanismFCAgent, env: SingleTurnEnvironment,
                    cond_key: str) -> Dict[str, Any]:
    """Evaluate one problem and return a result record."""
    out = agent.solve(env)
    gold = env.get_gold()

    ans_ok = answers_equal(out["pred_answer"], gold["answer"])

    # Evidence metrics: not meaningful for oracle_ev (we gave them the evidence)
    if agent.oracle_evidence:
        ev_p, ev_r, ev_f1 = 1.0, 1.0, 1.0
        ev_em = True
    else:
        ev_m = evidence_precision_recall_f1(
            out["pred_evidence_ids"], gold["evidence_ids"]
        )
        ev_p, ev_r, ev_f1 = ev_m["precision"], ev_m["recall"], ev_m["f1"]
        ev_em = evidence_exact_match(out["pred_evidence_ids"], gold["evidence_ids"])

    tool_calls = len(out.get("tool_trace", []))

    record = {
        "id": prob.id,
        "variant": prob.variant,
        "mode": cond_key,
        "pred_answer": out["pred_answer"],
        "gold_answer": gold["answer"],
        "answer_correct": ans_ok,
        "pred_evidence_ids": out["pred_evidence_ids"],
        "gold_evidence_ids": gold["evidence_ids"],
        "evidence_precision": ev_p,
        "evidence_recall": ev_r,
        "evidence_f1": ev_f1,
        "evidence_exact_match": ev_em,
        "n_chunks_seen": out["n_chunks_seen"],
        "tool_calls": tool_calls,
        "tool_trace": out.get("tool_trace", []),
        "expression": out.get("expression", ""),
        "reasoning": out.get("reasoning", ""),
        # Mechanism-specific flags for downstream analysis
        "tool_mode": agent.tool_mode,
        "oracle_evidence": agent.oracle_evidence,
        "max_tool_calls_limit": agent.max_tool_calls,
    }

    if getattr(agent, "g_step_gate", None) is not None or getattr(agent, "g_commit_gate", None) is not None:
        record.update(
            {
                "gate_action": out.get("gate_action", ""),
                "gate_p_continue": out.get("gate_p_continue", 0.0),
                "gate_p_one_more": out.get("gate_p_one_more", 0.0),
                "gate_anchor_applied": out.get("gate_anchor_applied", False),
                "gate_one_more_used": out.get("gate_one_more_used", 0),
                "gate_step_continue_count": out.get("gate_step_continue_count", 0),
                "gate_step_no_progress_blocks": out.get("gate_step_no_progress_blocks", 0),
                "gate_step_duplicate_blocks": out.get("gate_step_duplicate_blocks", 0),
                "gate_step_block_reasons": out.get("gate_step_block_reasons", {}),
            }
        )

    return record


# =========================================================================
# Main ablation runner
# =========================================================================

def run_mechanism_ablation(
    data_path: str,
    fc_model: str,
    fc_api_base: str,
    fc_api_key: str,
    variants: List[str] = None,
    limit: int = None,
    offset: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: int = 42,
    save_dir: str = None,
    sleep: float = 0.1,
    only: List[str] = None,
    g_step_model_path: Optional[str] = None,
    g_commit_model_path: Optional[str] = None,
    g_step_threshold: float = 0.05,
    g_commit_threshold: float = 0.5,
    g_step_max_extra_turns: int = 3,
    include_extended_gate_conditions: bool = False,
):
    """Run the mechanism ablation study."""

    # 1. Load data
    problems = load_problems(
        data_path,
        variants=variants,
        limit=limit,
        offset=offset,
        seed=seed,
    )
    groups = group_by_question(problems)
    print(f"Loaded {len(problems)} instances across {len(groups)} unique questions")
    print(f"Variants: {sorted(set(p.variant for p in problems))}")

    # 2. Init LLM
    llm = LLMBackend(
        model=fc_model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        api_base=fc_api_base,
        api_key=fc_api_key,
    )
    print(f"Model: {fc_model}  |  API: {fc_api_base}")

    g_step_gate = None
    g_commit_gate = None
    if g_step_model_path:
        g_step_gate = GStepGate.from_artifact(
            g_step_model_path,
            threshold=g_step_threshold,
            max_extra_turns=g_step_max_extra_turns,
        )
        print(
            "Loaded G_step gate: {0} (threshold={1})".format(
                g_step_model_path,
                g_step_threshold,
            )
        )
    if g_commit_model_path:
        g_commit_gate = GCommitGate.from_artifact(
            g_commit_model_path,
            threshold=g_commit_threshold,
        )
        print(
            "Loaded G_commit gate: {0} (threshold={1})".format(
                g_commit_model_path,
                g_commit_threshold,
            )
        )

    # 3. Condition definitions
    #    Each entry: (key, description, agent_kwargs)
    ALL_CONDITIONS = [
        (
            "fc_baseline",
            f"FC-Baseline (real calculator, all chunks, max 5 calls, forced first tool, {fc_model})",
            {
                "tool_mode": "real",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_auto",
            f"FC-AutoTool (real calculator, all chunks, max 5 calls, auto first turn, {fc_model})",
            {
                "tool_mode": "real",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": False,
            },
        ),
        (
            "fc_noop",
            f"FC-NoOp (tool echoes expression, no computation — tests protocol effect)",
            {
                "tool_mode": "noop",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_auto_noop",
            f"FC-AutoNoOp (NoOp tool, all chunks, max 5 calls, auto first turn, {fc_model})",
            {
                "tool_mode": "noop",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": False,
            },
        ),
        (
            "fc_perfect",
            f"FC-Perfect (tool returns gold answer — computation upper bound)",
            {
                "tool_mode": "perfect",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_max1",
            f"FC-Max1 (real calculator, max 1 call — tests value of iteration)",
            {
                "tool_mode": "real",
                "max_tool_calls": 1,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_oracle_ev",
            f"FC-OracleEv (real calculator, only evidence chunks — tests H1)",
            {
                "tool_mode": "real",
                "max_tool_calls": 5,
                "oracle_evidence": True,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_notool_cot",
            f"FC-NoTool-CoT (no tools, CoT prompt — tests CoT prompt design)",
            {
                "tool_mode": "notool",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
        (
            "fc_notool_fcprompt",
            f"FC-NoTool-FCPrompt (no tools, same FC prompt as baseline — isolates protocol overhead)",
            {
                "tool_mode": "fcprompt",
                "max_tool_calls": 5,
                "oracle_evidence": False,
                "force_tool_first_turn": True,
            },
        ),
    ]

    if g_step_gate is not None:
        ALL_CONDITIONS.extend(
            [
                (
                    "gate_step",
                    "G-STEP (paper gate with original continuation prompt)",
                    {
                        "tool_mode": "real",
                        "max_tool_calls": 5,
                        "oracle_evidence": False,
                        "force_tool_first_turn": True,
                        "g_step_gate": g_step_gate,
                        "g_step_max_extra_turns": g_step_max_extra_turns,
                        "g_step_continue_prompt": DEFAULT_GSTEP_CONTINUE_PROMPT,
                        "g_step_replan_prompt": DEFAULT_GSTEP_REPLAN_PROMPT,
                        "g_step_duplicate_replan_prompt": DEFAULT_GSTEP_DUPLICATE_REPLAN_PROMPT,
                        "gate_force_tool_on_continue": True,
                    },
                ),
                (
                    "gate_step_critic",
                    "G-STEP+CRITIC (paper gate with critic-style continuation prompts)",
                    {
                        "tool_mode": "real",
                        "max_tool_calls": 5,
                        "oracle_evidence": False,
                        "force_tool_first_turn": True,
                        "g_step_gate": g_step_gate,
                        "g_step_max_extra_turns": g_step_max_extra_turns,
                        "g_step_continue_prompt": CRITIC_GSTEP_CONTINUE_PROMPT,
                        "g_step_replan_prompt": CRITIC_GSTEP_REPLAN_PROMPT,
                        "g_step_duplicate_replan_prompt": CRITIC_GSTEP_DUPLICATE_REPLAN_PROMPT,
                        "g_commit_one_more_prompt": CRITIC_GCOMMIT_ONE_MORE_PROMPT,
                        "gate_force_tool_on_continue": True,
                    },
                ),
            ]
        )

    if include_extended_gate_conditions and g_commit_gate is not None:
        ALL_CONDITIONS.append(
            (
                "gate_commit_only",
                "Gate-CommitOnly (FC + G_commit; one_more before submit)",
                {
                    "tool_mode": "real",
                    "max_tool_calls": 5,
                    "oracle_evidence": False,
                    "force_tool_first_turn": True,
                    "g_step_gate": None,
                    "g_commit_gate": g_commit_gate,
                    "g_step_max_extra_turns": g_step_max_extra_turns,
                },
            )
        )
    if include_extended_gate_conditions and (g_step_gate is not None) and (g_commit_gate is not None):
        ALL_CONDITIONS.append(
            (
                "gate_step_commit",
                "Gate-StepCommit (FC + G_step + G_commit)",
                {
                    "tool_mode": "real",
                    "max_tool_calls": 5,
                    "oracle_evidence": False,
                    "force_tool_first_turn": True,
                    "g_step_gate": g_step_gate,
                    "g_commit_gate": g_commit_gate,
                    "g_step_max_extra_turns": g_step_max_extra_turns,
                },
            )
        )

    # Filter to specific conditions if --only
    if only:
        missing = [key for key in only if key not in {condition[0] for condition in ALL_CONDITIONS}]
        if missing:
            raise ValueError(
                "Requested conditions unavailable. Missing: {0}. If these are gate "
                "conditions, provide --g_step_model_path / --g_commit_model_path, and "
                "use --include_extended_gate_conditions for legacy g_commit conditions.".format(
                    ",".join(missing)
                )
            )
        CONDITIONS = [(k, d, kw) for k, d, kw in ALL_CONDITIONS if k in only]
        print(f"Running only: {[k for k,_,_ in CONDITIONS]}")
    else:
        CONDITIONS = ALL_CONDITIONS

    # 4. Env config
    st_cfg = SingleTurnEnvConfig(top_k=999, shuffle_chunks=False, seed=seed)

    all_condition_results: Dict[str, List[Dict]] = {}

    # 5. Run each condition
    for cond_key, cond_desc, agent_kwargs in CONDITIONS:
        print(f"\n{'='*70}")
        print(f"  {cond_desc}")
        print(f"{'='*70}")

        agent = MechanismFCAgent(llm=llm, **agent_kwargs)
        results = []

        for prob in tqdm(problems, desc=cond_key):
            env = SingleTurnEnvironment(prob, st_cfg)
            r = _eval_mechanism(prob, agent, env, cond_key)
            results.append(r)
            if sleep > 0:
                time.sleep(sleep)

        all_condition_results[cond_key] = results
        # Quick per-condition summary
        acc = sum(r["answer_correct"] for r in results) / len(results)
        avg_tc = sum(r["tool_calls"] for r in results) / len(results)
        print(f"  → acc={acc:.3f}  avg_tool_calls={avg_tc:.2f}  n={len(results)}")

    # 6. Print comparison table
    _print_mechanism_table(all_condition_results)

    # 7. Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        all_flat = []
        for cond_key, results in all_condition_results.items():
            path = os.path.join(save_dir, f"{cond_key}.jsonl")
            # Merge with existing file if --only was used
            existing = []
            if only and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    existing = [json.loads(l) for l in f if l.strip()]
                # Replace old records for the same (id, variant) so re-runs don't duplicate.
                new_keys = {(r.get("id"), r.get("variant")) for r in results}
                existing = [
                    r for r in existing
                    if (r.get("id"), r.get("variant")) not in new_keys
                ]
            with open(path, "w", encoding="utf-8") as w:
                for r in existing + results:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_flat.extend(results)

        # Combined file (merge with existing if --only)
        all_path = os.path.join(save_dir, "all_results.jsonl")
        existing_all = []
        if only and os.path.exists(all_path):
            with open(all_path, encoding="utf-8") as f:
                existing_all = [json.loads(l) for l in f if l.strip()]
            # Remove old records for conditions being re-run
            existing_all = [r for r in existing_all if r.get("mode") not in only]
        with open(all_path, "w", encoding="utf-8") as w:
            for r in existing_all + all_flat:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary report
        summary = _build_summary(all_condition_results)
        with open(os.path.join(save_dir, "mechanism_report.json"), "w", encoding="utf-8") as w:
            json.dump(summary, w, indent=2, ensure_ascii=False)

        print(f"\nSaved to {save_dir}/")

    return all_condition_results


# =========================================================================
# Reporting helpers
# =========================================================================

def _build_summary(all_condition_results: Dict[str, List[Dict]]) -> Dict:
    """Build a nested summary dict: {cond_key: {variant: {acc, avg_tc, ...}}}."""
    summary = {}
    for cond_key, results in all_condition_results.items():
        by_var: Dict[str, Dict] = {}
        for r in results:
            var = r["variant"]
            if var not in by_var:
                by_var[var] = {"correct": [], "tool_calls": [], "evidence_f1": []}
            by_var[var]["correct"].append(r["answer_correct"])
            by_var[var]["tool_calls"].append(r["tool_calls"])
            by_var[var]["evidence_f1"].append(r["evidence_f1"])

        cond_summary = {}
        all_correct = []
        for var, d in sorted(by_var.items()):
            n = len(d["correct"])
            acc = sum(d["correct"]) / n
            avg_tc = sum(d["tool_calls"]) / n
            avg_f1 = sum(d["evidence_f1"]) / n
            cond_summary[var] = {
                "acc": round(acc, 4),
                "avg_tool_calls": round(avg_tc, 3),
                "evidence_f1": round(avg_f1, 4),
                "n": n,
            }
            all_correct.extend(d["correct"])
        cond_summary["_overall"] = {
            "acc": round(sum(all_correct) / len(all_correct), 4),
            "n": len(all_correct),
        }
        summary[cond_key] = cond_summary
    return summary


def _print_mechanism_table(all_condition_results: Dict[str, List[Dict]]):
    """Print a focused mechanism ablation comparison table."""
    print("\n" + "=" * 100)
    print("MECHANISM ABLATION — What drives the FC benefit?")
    print("=" * 100)

    variants = sorted(set(
        r["variant"]
        for results in all_condition_results.values()
        for r in results
    ))

    COND_ORDER = [
        "fc_baseline",
        "fc_auto",
        "fc_noop",
        "fc_auto_noop",
        "fc_perfect",
        "fc_max1",
        "fc_oracle_ev",
        "gate_step",
        "gate_step_critic",
        "gate_commit_only",
        "gate_step_commit",
    ]
    COND_LABELS = {
        "fc_baseline":  "FC-Baseline ",
        "fc_auto":      "FC-AutoTool ",
        "fc_noop":      "FC-NoOp     ",
        "fc_auto_noop": "FC-AutoNoOp ",
        "fc_perfect":   "FC-Perfect  ",
        "fc_max1":      "FC-Max1     ",
        "fc_oracle_ev": "FC-OracleEv ",
        "gate_step": "G-STEP      ",
        "gate_step_critic": "G-STEP+CRT ",
        "gate_commit_only": "Gate-Commit ",
        "gate_step_commit": "Gate-Step+Co",
    }

    # Per-condition accuracy by variant
    cond_stats: Dict[str, Dict[str, Dict]] = {}
    for cond_key, results in all_condition_results.items():
        by_var: Dict[str, Dict] = {}
        for r in results:
            var = r["variant"]
            if var not in by_var:
                by_var[var] = {"correct": [], "tool_calls": []}
            by_var[var]["correct"].append(r["answer_correct"])
            by_var[var]["tool_calls"].append(r["tool_calls"])
        cond_stats[cond_key] = by_var

    # Header
    print(f"\n{'Condition':<14}", end="")
    for var in variants:
        print(f" │ {var:^17}", end="")
    print(f" │ {'overall':^8}")
    print("-" * (14 + 20 * len(variants) + 12))

    for cond_key in COND_ORDER:
        if cond_key not in cond_stats:
            continue
        label = COND_LABELS.get(cond_key, cond_key)
        print(f"{label:<14}", end="")
        all_correct = []
        for var in variants:
            if var in cond_stats[cond_key]:
                s = cond_stats[cond_key][var]
                n = len(s["correct"])
                acc = sum(s["correct"]) / n * 100
                avg_tc = sum(s["tool_calls"]) / n
                all_correct.extend(s["correct"])
                print(f" │ {acc:5.1f}% tc={avg_tc:.2f}  ", end="")
            else:
                print(f" │ {'---':^17}", end="")
        if all_correct:
            ov = sum(all_correct) / len(all_correct) * 100
            print(f" │ {ov:5.1f}%")
        else:
            print()

    # Delta table: all conditions vs fc_baseline
    if "fc_baseline" in cond_stats:
        print("\n" + "-" * 100)
        print("DELTAS (percentage points; mostly vs fc_baseline, NoOp policy vs fc_noop):")
        print("-" * 100)

        COMPARISONS = [
            ("fc_auto",
             "fc_baseline vs fc_auto      [policy: force/auto]"),
            ("fc_noop",
             "fc_baseline vs fc_noop      [H3: computation]  "),
            ("fc_auto_noop",
             "fc_noop vs fc_auto_noop     [policy under NoOp]"),
            ("fc_perfect",
             "fc_baseline vs fc_perfect   [ceiling test]     "),
            ("fc_max1",
             "fc_baseline vs fc_max1      [H2: iteration]    "),
            ("fc_oracle_ev",
             "fc_baseline vs fc_oracle_ev [H1: evidence sel] "),
            ("gate_step",
             "fc_baseline vs g_step       [paper gate]         "),
            ("gate_step_critic",
             "fc_baseline vs g_step+critic[prompted gate]      "),
            ("gate_commit_only",
             "fc_baseline vs gate_commit  [commit gate gain] "),
            ("gate_step_commit",
             "fc_baseline vs gate_step+co [step+commit gain] "),
        ]

        base = cond_stats["fc_baseline"]
        noop = cond_stats.get("fc_noop")
        for other_key, label in COMPARISONS:
            if other_key not in cond_stats:
                continue
            other = cond_stats[other_key]
            deltas = []
            for var in variants:
                if var not in other:
                    continue
                if other_key == "fc_auto_noop":
                    if noop is None or var not in noop:
                        continue
                    acc_base = sum(noop[var]["correct"]) / len(noop[var]["correct"]) * 100
                else:
                    if var not in base:
                        continue
                    acc_base = sum(base[var]["correct"]) / len(base[var]["correct"]) * 100
                acc_other = sum(other[var]["correct"]) / len(other[var]["correct"]) * 100
                delta = acc_other - acc_base
                deltas.append(f"{var}:{delta:+.1f}")
            print(f"  {label}  {',  '.join(deltas)}")

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("  fc_baseline vs fc_auto:     Gap = cost/benefit of forcing first tool call")
    print("  fc_baseline vs fc_noop:     SMALL gap → protocol effect dominates (not computation)")
    print("                              LARGE gap → actual computation is the key driver")
    print("  fc_noop vs fc_auto_noop:    Gap = forced-call overhead under NoOp tools")
    print("  fc_baseline vs fc_perfect:  Gap = improvement possible with perfect computation")
    print("  fc_baseline vs fc_max1:     Gap = value of iterative tool calls (key for SP)")
    print("  fc_baseline vs fc_oracle_ev:Gap = accuracy lost due to noise in evidence selection")
    if "gate_step" in all_condition_results:
        print("  fc_baseline vs g_step:      Gain from the paper's G-STEP intervention")
    if "gate_step_critic" in all_condition_results:
        print("  fc_baseline vs g_step+critic:Gain from critic-style continuation prompting")
    if "gate_commit_only" in all_condition_results:
        print("  fc_baseline vs gate_commit: Gain from pre-submit one_more gate")
    if "gate_step_commit" in all_condition_results:
        print("  fc_baseline vs gate_step+co:Gain from step-level continue + commit gate")
    print("  tc = avg tool calls")
    print("=" * 100)


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Mechanism ablation: why does real FC improve performance?"
    )
    ap.add_argument("--data", required=True,
                    help="Path to pipeline output JSONL (e.g. train_aug_150_f11b11.jsonl)")
    ap.add_argument("--fc_model", required=True,
                    help="Model name (e.g. gpt-4.1-mini)")
    ap.add_argument("--fc_api_base", default="https://api.openai.com/v1",
                    help="API base URL")
    ap.add_argument("--fc_api_key", default=None,
                    help="API key (or set OPENAI_API_KEY env var)")
    ap.add_argument("--variants", default=None,
                    help="Comma-separated variants, e.g. base,TB,PED,HU,SP")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max unique question IDs per variant")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N unique question IDs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048,
                    help="Max output tokens per API call")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default=None,
                    help="Directory to save per-condition JSONL + mechanism_report.json")
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="Sleep between API calls (seconds) to avoid rate limits")
    ap.add_argument("--only", default=None,
                    help="Comma-separated condition keys to run "
                         "(e.g. fc_noop,fc_oracle_ev). Omit to run all.")
    ap.add_argument("--g_step_model_path", default=None,
                    help="Path to a trained G_step artifact (.pkl or .json)")
    ap.add_argument("--g_commit_model_path", default=None,
                    help="Path to a trained G_commit artifact (.pkl or .json; non-paper extension)")
    ap.add_argument("--g_step_threshold", type=float, default=0.05,
                    help="Decision threshold for the G_step continue action")
    ap.add_argument("--g_commit_threshold", type=float, default=0.5,
                    help="Decision threshold for the G_commit one_more action")
    ap.add_argument("--g_step_max_extra_turns", type=int, default=3,
                    help="Maximum extra continuation turns triggered by G_step")
    ap.add_argument("--include_extended_gate_conditions", action="store_true",
                    help="Also expose legacy non-paper G_commit conditions when commit artifacts are provided.")

    args = ap.parse_args()

    is_mock_backend = args.fc_model.startswith("mock") or args.fc_api_base.startswith("mock://")

    api_key = args.fc_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and (
        "gemini" in args.fc_model.lower()
        or "googleapis.com" in args.fc_api_base.lower()
        or "generativelanguage" in args.fc_api_base.lower()
    ):
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not is_mock_backend:
        ap.error("Provide --fc_api_key or set OPENAI_API_KEY (or GEMINI_API_KEY) environment variable")

    variants = args.variants.split(",") if args.variants else None
    only = args.only.split(",") if args.only else None

    run_mechanism_ablation(
        data_path=args.data,
        fc_model=args.fc_model,
        fc_api_base=args.fc_api_base,
        fc_api_key=api_key,
        variants=variants,
        limit=args.limit,
        offset=args.offset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        save_dir=args.save_dir,
        sleep=args.sleep,
        only=only,
        g_step_model_path=args.g_step_model_path,
        g_commit_model_path=args.g_commit_model_path,
        g_step_threshold=args.g_step_threshold,
        g_commit_threshold=args.g_commit_threshold,
        g_step_max_extra_turns=args.g_step_max_extra_turns,
        include_extended_gate_conditions=args.include_extended_gate_conditions,
    )


if __name__ == "__main__":
    main()
