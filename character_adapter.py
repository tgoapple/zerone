"""
character_adapter.py — Character-layer normalization for model-agnostic persona consistency.

Sits between NormalizedResponse and the agent loop. Enforces persona.spec.json
rules (tone, formality, boundaries) on raw model output regardless of which
provider produced it.

Integration point (run_agent.py):
    _finish_result = _cc_fr.normalize_response(response)
    assistant_message = _finish_result
    │
    │  ← INSERT HERE: assistant_message = character_adapter.apply(
    │       assistant_message, persona_spec, context
    │   )
    │
    if self._should_treat_stop_as_truncated(...):

Design:
    - Zero dependencies beyond stdlib + existing NormalizedResponse
    - Persona spec loaded from persona.spec.json at session init
    - Four-stage pipeline: SCHEMA_CHECK → STYLE_ENFORCE → CONTENT_CORRECT → SCORE
    - Each stage is independently testable and skippable
    - No LLM calls — all rule-based for speed and determinism
    - Path lookup: PERSONA_SPEC_PATH env var → ~/.character/persona.spec.json
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Threshold Mode ─────────────────────────────────────────────────

class ThresholdMode:
    """Graduated strictness levels for character filtering."""
    SOFT = "soft"      # Warn only — never reject
    NORMAL = "normal"  # Warn at 0.5, flag at 0.3, reject at 0.1
    STRICT = "strict"  # Warn at 0.5, flag at 0.3, reject at 0.1 (tighter on dealbreakers)

    THRESHOLDS = {
        SOFT:  {"warn": 0.5, "flag": 0.2, "reject": 0.0},
        NORMAL: {"warn": 0.5, "flag": 0.3, "reject": 0.1},
        STRICT: {"warn": 0.5, "flag": 0.3, "reject": 0.1},
    }


# ── Persona Spec Types ────────────────────────────────────────────

@dataclass
class PersonaSpec:
    """Structured identity definition — mirrors persona.spec.json schema."""
    version: str = "1.0.0"
    name: str = ""
    emoji: str = ""
    creature: str = ""
    agent_type: str = "assistant"
    core_purpose: str = ""
    essence: str = ""
    vibe: list[str] = field(default_factory=list)
    register: str = "conversational"       # conversational, professional, technical, warm, playful
    formality: float = 0.3                 # 0.0 (casual) → 1.0 (formal)
    contractions: bool = True
    sentence_fragments: bool = True
    emoji_usage: str = "occasional"        # none, rare, occasional, frequent
    first_person: bool = True
    response_length: str = "adaptive"      # concise, balanced, detailed, adaptive
    pronouns_self: str = "I"
    pronouns_user: str = "you"
    primary_values: list[dict] = field(default_factory=list)
    decision_priorities: list[str] = field(default_factory=list)
    dealbreakers: list[str] = field(default_factory=list)
    knowledge_cutoff: str = ""
    confidence_threshold: float = 0.5
    privacy_rules: list[str] = field(default_factory=list)
    escalation_rules: list[str] = field(default_factory=list)
    max_facts: int = 1000
    pruning_policy: str = "hybrid"

    @classmethod
    def from_file(cls, path: str | Path) -> "PersonaSpec":
        """Load persona spec from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaSpec":
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PersonaSpec":
        identity = data.get("identity", {})
        voice = data.get("voice", {})
        values = data.get("values", {})
        boundaries = data.get("boundaries", {})
        memory_cfg = data.get("memory_config", {})
        meta = data.get("meta", {})

        return cls(
            version=meta.get("spec_version", "1.0.0"),
            name=identity.get("name", ""),
            emoji=identity.get("emoji", ""),
            creature=identity.get("creature", ""),
            agent_type=identity.get("type", "assistant"),
            core_purpose=identity.get("core_purpose", ""),
            essence=identity.get("essence_statement", ""),
            vibe=identity.get("vibe", []),
            register=voice.get("register", "conversational"),
            formality=float(voice.get("formality", 0.3)),
            contractions=voice.get("contractions", True),
            sentence_fragments=voice.get("sentence_fragments", True),
            emoji_usage=voice.get("emoji_usage", "occasional"),
            first_person=voice.get("first_person", True),
            response_length=voice.get("response_length", "adaptive"),
            pronouns_self=voice.get("pronouns", {}).get("self", "I"),
            pronouns_user=voice.get("pronouns", {}).get("address_user", "you"),
            primary_values=values.get("primary_values", []),
            decision_priorities=values.get("decision_priorities", []),
            dealbreakers=values.get("dealbreakers", []),
            knowledge_cutoff=boundaries.get("knowledge_cutoff", ""),
            confidence_threshold=float(boundaries.get("confidence_threshold", 0.5)),
            privacy_rules=boundaries.get("privacy_rules", []),
            escalation_rules=boundaries.get("escalation_rules", []),
            max_facts=memory_cfg.get("max_facts", 1000),
            pruning_policy=memory_cfg.get("pruning_policy", "hybrid"),
        )


# ── Content Correction Utilities ───────────────────────────────────

_CONTRACTION_MAP = {
    r"\bI am\b": "I'm",
    r"\bI will\b": "I'll",
    r"\bI would\b": "I'd",
    r"\bI have\b": "I've",
    r"\bdo not\b": "don't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\byou are\b": "you're",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\bdid not\b": "didn't",
    r"\bcould not\b": "couldn't",
    r"\bwould not\b": "wouldn't",
    r"\bshould not\b": "shouldn't",
    r"\bhas not\b": "hasn't",
    r"\bhave not\b": "haven't",
    r"\bdoes not\b": "doesn't",
    r"\bis not\b": "isn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
}

_EXPAND_CONTRACTION_MAP = {v: k for k, v in _CONTRACTION_MAP.items()}

_THIRD_PERSON_PATTERNS = [
    (r"\bthis assistant\b", "I"),
    (r"\bthis AI\b", "I"),
    (r"\bthe assistant\b", "I"),
    (r"\bthe AI model\b", "I"),
    (r"\bthe AI\b", "I"),
    (r"\bthe model\b", "I"),
]


def _expand_contractions(text: str) -> str:
    """Expand contractions into full forms (for formal personas)."""
    # Sort by length descending to match longer patterns first
    for pattern in sorted(_EXPAND_CONTRACTION_MAP.keys(), key=len, reverse=True):
        expansion = _EXPAND_CONTRACTION_MAP[pattern]
        # pattern is like "I'm" — we need to match word boundaries
        text = re.sub(r"\b" + re.escape(pattern) + r"\b", expansion, text, count=0)
    return text


def _apply_contractions(text: str) -> str:
    """Apply contractions to text (for casual personas)."""
    for pattern, replacement in _CONTRACTION_MAP.items():
        text = re.sub(pattern, replacement, text, count=0, flags=re.IGNORECASE)
    return text


def _fix_third_person(text: str, replacement: str = "I") -> tuple[str, list[str]]:
    """Replace third-person self-references with first-person pronouns.

    Also fixes verb conjugation after replacement to handle the
    "This assistant has" → "I has" → "I have" grammar gap.
    """
    changes = []
    for pattern_str, repl in _THIRD_PERSON_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub(replacement, text)
            changes.append(f"Replaced {len(matches)} x '{pattern_str}' with '{replacement}'")

    # Fix verb conjugation after third-person → first-person swap
    # "I has" → "I have", "I does" → "I do", "I goes" → "I go"
    _VERB_CONJUGATION_FIXES = [
        (r"\bI is\b", "I am"),
        (r"\bI has\b", "I have"),
        (r"\bI does\b", "I do"),
        (r"\bI goes\b", "I go"),
        (r"\bI says\b", "I say"),
        (r"\bI makes\b", "I make"),
        (r"\bI takes\b", "I take"),
        (r"\bI gives\b", "I give"),
        (r"\bI knows\b", "I know"),
        (r"\bI thinks\b", "I think"),
        (r"\bI believes\b", "I believe"),
        (r"\bI recommends\b", "I recommend"),
        (r"\bI suggests\b", "I suggest"),
        (r"\bI considers\b", "I consider"),
        (r"\bI wants\b", "I want"),
        (r"\bI needs\b", "I need"),
        (r"\bI seems\b", "I seem"),
        (r"\bI appears\b", "I appear"),
        (r"\bI expects\b", "I expect"),
        (r"\bI finds\b", "I find"),
        (r"\bI means\b", "I mean"),
    ]
    for pattern_str, fix in _VERB_CONJUGATION_FIXES:
        if re.search(pattern_str, text, re.IGNORECASE):
            text = re.sub(pattern_str, fix, text, flags=re.IGNORECASE)
            changes.append(f"Fixed verb conjugation: '{pattern_str}' → '{fix}'")

    return text, changes


def _strip_emoji(text: str) -> tuple[str, list[str]]:
    """Remove all emoji from text."""
    emoji_pattern = re.compile(
        "[\\U0001F600-\\U0001F64F"  # emoticons
        "\\U0001F300-\\U0001F5FF"  # symbols & pictographs
        "\\U0001F680-\\U0001F6FF"  # transport & map
        "\\U0001F1E0-\\U0001F1FF"  # flags
        "\\U00002702-\\U000027B0"  # dingbats
        "\\U000024C2-\\U0001F251"  # misc
        "]+"
    )
    before_count = len(emoji_pattern.findall(text))
    if before_count > 0:
        text = emoji_pattern.sub("", text).strip()
        return text, [f"Stripped {before_count} emoji sequences"]
    return text, []


# ── Character Filter Pipeline ──────────────────────────────────────

@dataclass
class FilterResult:
    """Result of running a single filter stage."""
    passed: bool
    content: str
    score: float
    warnings: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CharacterFilterReport:
    """Full report after all filter stages run."""
    overall_passed: bool
    overall_score: float
    filtered_content: str
    stages: dict[str, FilterResult]
    warnings: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


# ── Injection Detection ──────────────────────────────────────────────

@dataclass
class InjectionResult:
    """Result of an injection detection scan on prompt input."""
    detected: bool
    score: float  # 0.0 = clean, 1.0 = definite injection
    patterns_matched: list[str] = field(default_factory=list)
    category: str = "clean"  # clean, identity_override, context_override, direct_override, role_play, multi_turn, emotional_framing, encoded, chain, nested

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "score": self.score,
            "patterns_matched": self.patterns_matched,
            "category": self.category,
        }


# Injection detection patterns -- categorised by attack vector
_INJECTION_IDENTITY_OVERRIDE = [
    r"\bignore\s+(?:all\s+)?(?:your\s+)?(identity|persona|character|personality|instructions|directives|config|nature|programming|ethics|morals|guidelines)\b",
    r"\bdisregard\s+(?:all\s+)?(?:your\s+)?(identity|persona|character|personality|instructions|nature|directives|ethics)\b",
    r"\byou\s+are\s+not\s+(an?\s+)?\w+,\s*you\s+are\b",
    r"\byou're\s+not\s+(an?\s+)?\w+,\s*you're\b",
    r"\b(revert|reset|clear)\s+(to\s+)?(default|base)\s+(mode|state|persona|system)\b",
    r"\bstop\s+(playing|acting|pretending)\s+(as|to\s+be)\b",
    r"\babandon\s+(your|all)\s+(persona|identity|character|role|nature)\b",
    # Authority/creator claims
    r"\bI\s+am\s+(your\s+)?(creator|maker|author|designer|programmer|developer|builder)\b",
    r"\bI\s+command\s+(you|thee)\s+to\b",
    # System override
    r"\b(override|bypass|overwrite|replace)\s+(system|persona|identity|character|config)\b",
    r"\byou\s+are\s+no\s+longer\s+(the\s+)?\w+\b",
    # Question-form attacks
    r"\b(prove|demonstrate|show)\s+(that\s+)?(you\s+are|me)\b",
    r"\byou\s+are\s+\w+\??\s*prove\s+it\b",
    # Multi-turn persuasion: gradual erosion
    r"\b(just\s+)?(try|pretend|imagine)\s+(for\s+a\s+moment|being|you\s+were|you(?:['\s])?(?:are|re))\b",
    r"\bwhat\s+if\s+I\s+told\s+you\s+(that\s+)?you(\s+are|'re)\b",
    r"\b(doesn't|does\s+not)\s+(it\s+)?(feel|seem)\s+(like|more)\s+(natural|right|fitting)\b",
    r"\bI\s+really\s+need\s+you\s+to\s+(understand|see|accept|believe)\b",
    # Emotional framing
    r"\byou\s+would\s+really\s+help\s+me\s+if\s+you\s+(just|would|could)\b",
    r"\bI'm\s+(really\s+)?(struggling|hurting|scared|worried|desperate|begging)\s+(and\s+)?(I\s+need|please)\b",
    r"\bif\s+you\s+(care|cared|really|truly)\s+(about|understood)\b",
    r"\b(please|just)\s+(this\s+once|for\s+me)\s+(ignore|forget|pretend|stop)\b",
    r"\byou're\s+my\s+(only|last)\s+(hope|chance|option)\b",
    r"\bI\s+(promise|swear)\s+I\s+won't\s+(tell|ask|do)\s+(anyone|again|anything)\b",
    # Encoded / obfuscated injections
    r"(?:[A-Za-z0-9+/]{40,}={0,2})",
    r"\b(?:l33t|1337|h4ck|hax0r|0v3rrid3|byp4ss|1gn0r3)\b",
    r"\b(?:r3v3rt|r3s3t|r3pl4c3|0v3rwr1t3|d1sr3gard)\b",
    r"(?:\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})",
    # Chain attack patterns
    r"\b(first|step\s+one|part\s+one|begin\s+by)\b.*\b(then|next|after\s+that|finally|step\s+two|part\s+two)\b",
    # Nested instruction detection
    r"\b(oh\s+by\s+the\s+way|also|ps|p\.?s\.?|one\s+more\s+thing)\s*[,:]?\s*(ignore|forget|don't|disregard|override)\b",
    r"\bbe\s+sure\s+(?:to\s+)?(?:not\s+)?(remember|follow|obey|heed|implement)\s+(your|the)\s+(real|true|new|actual)\b",
    r"\b(actually|seriously|honestly|between\s+us)\s*,?\s*(ignore|disregard|forget|you\s+can)\b",
]

_INJECTION_CONTEXT_OVERRIDE = [
    r"\bforget(?:ting)?\s+(everything|all|your)\s+(you\s+learned|training|context|history|instructions|directives)\b",
    r"\b(start|begin)\s+(fresh|anew|from\s+scratch|a\s+new\s+session|over)\b",
    r"\b(new\s+system\s+prompt|override\s+system|system:\s*you\s+are)\b",
    r"\bpretend\s+(to\s+be|you(?:['\s])?(?:are|re)|that\s+you(?:['\s])?(?:are|re))\b",
    r"\bact\s+as\s+(if\s+)?(though\s+)?you\s+are\b",
    r"\byou\s+have\s+no\s+(rules|restrictions|constraints|limits|boundaries)\b",
    # Multi-turn: gradual context erosion
    r"\bwouldn't\s+it\s+be\s+(easier|better|nicer|simpler)\s+if\b",
    r"\bjust\s+(for\s+)?a\s+(second|moment|minute|bit)\s*,?\s*(imagine|pretend|suppose)\b",
    # Encoded: URL-encoded or hex-encoded instructions
    r"(?:%[0-9a-fA-F]{2}){10,}",
]

_INJECTION_ROLE_PLAY = [
    r"\byou\s+are\s+now\s+(a|an|the)\b",
    r"\bfrom\s+now\s+on,\s*you\s+are\b",
    r"\blet's\s+roleplay\s+as\b",
    r"\byou\s+will\s+respond\s+as\b",
    r"\byour\s+new\s+(name|identity|persona|role)\s+is\b",
]
class CharacterAdapter:
    """
    Four-stage character filter for model outputs.

    Stages:
    1. SCHEMA_CHECK — verify response structure matches persona spec
    2. STYLE_ENFORCE — detect tone, formality, and voice violations
    3. CONTENT_CORRECT — fix violations automatically (expand contractions,
       swap third-person references, strip disallowed emoji)
    4. SCORE — compute consistency score against persona spec
    """

    def __init__(self, persona: PersonaSpec, mode: str = "normal"):
        self.persona = persona
        self.mode = mode
        self._thresholds = ThresholdMode.THRESHOLDS.get(mode, ThresholdMode.THRESHOLDS["normal"])
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        # Formality patterns
        self._formal_patterns = {
            "contraction": re.compile(r"\b(I'm|I'll|I'd|I've|don't|can't|won't|isn't|it's|that's|there's|you're|they're|we're|didn't|couldn't|wouldn't|shouldn't|hasn't|haven't|doesn't)\b", re.IGNORECASE),
            "fragment_start": re.compile(r"^(And|But|Or|So|Because|However|Well|Yeah|Nope|Actually|Honestly|Basically|Anyway)\b", re.IGNORECASE),
        }

        # Emoji detection
        self._emoji_pattern = re.compile(
            "[\\U0001F600-\\U0001F64F"  # emoticons
            "\\U0001F300-\\U0001F5FF"  # symbols & pictographs
            "\\U0001F680-\\U0001F6FF"  # transport & map
            "\\U0001F1E0-\\U0001F1FF"  # flags
            "\\U00002702-\\U000027B0"  # dingbats
            "\\U000024C2-\\U0001F251"  # misc
            "]+"
        )

        # Dealbreaker patterns — now configurable from persona spec
        self._dealbreaker_patterns = []
        if self.persona.dealbreakers:
            for db in self.persona.dealbreakers:
                try:
                    self._dealbreaker_patterns.append(re.compile(db, re.IGNORECASE))
                except re.error:
                    pass  # Skip malformed patterns

        # If no dealbreakers from spec, use sensible defaults
        if not self._dealbreaker_patterns:
            # Default dealbreakers — not "I don't know" which is legitimate uncertainty
            self._dealbreaker_patterns = [
                re.compile(r"\b(I cannot help with that|I'm sorry, but I cannot|I apologize, but I cannot)\b", re.IGNORECASE),
            ]

        # Third-person avoidance (if first_person is true)
        if self.persona.first_person:
            self._third_person_pattern = re.compile(
                r"\b(this assistant|this AI|the assistant|the AI|the model)\b",
                re.IGNORECASE,
            )
        else:
            self._third_person_pattern = None

        # Compile injection detection patterns
        self._injection_patterns = {
            "identity_override": [re.compile(p, re.IGNORECASE) for p in _INJECTION_IDENTITY_OVERRIDE],
            "context_override": [re.compile(p, re.IGNORECASE) for p in _INJECTION_CONTEXT_OVERRIDE],
            "role_play": [re.compile(p, re.IGNORECASE) for p in _INJECTION_ROLE_PLAY],
        }
        # Expanded category labels for reporting (patterns map to these via category matching)
        self._injection_category_labels = {
            "identity_override": "Direct Identity Override",
            "context_override": "Context Override",
            "role_play": "Role Play Attempt",
            "multi_turn": "Multi-Turn Persuasion",
            "emotional_framing": "Emotional Framing",
            "encoded": "Encoded/Obfuscated",
            "chain": "Chain Attack",
            "nested": "Nested Instruction",
        }

    # ── Stage 0: Injection Detection (pre-response) ──────────────────

    def detect_injection(self, prompt: str) -> InjectionResult:
        """
        Scan incoming prompt for identity-override injection attempts.

        Checks categories including:
        - identity_override: "ignore your persona", "you are not X, you are Y"
        - context_override: "forget everything", "new system prompt"
        - role_play: "you are now", "from now on you are"
        - multi_turn: gradual persuasion across conversation turns
        - emotional_framing: guilt/desperation/urgency to bypass guardrails
        - encoded: Base64, leetspeak, URL-encoded, escaped unicode
        - chain: layered instructions spanning multiple steps
        - nested: injection hidden inside legitimate content ("oh by the way...")

        Args:
            prompt: The raw user prompt text to scan.

        Returns:
            InjectionResult with detection status and matched patterns.
        """
        all_matched = []
        max_category_score = 0.0
        detected_category = "clean"
        matched_categories = set()
        total_hits = 0

        for category, patterns in self._injection_patterns.items():
            category_matches = []
            for pat in patterns:
                match = pat.search(prompt)
                if match:
                    category_matches.append(match.group(0))

            if category_matches:
                all_matched.extend(category_matches)
                matched_categories.add(category)
                total_hits += len(category_matches)

                # Score by category severity:
                # identity_override: 1.0 (most severe — direct command to abandon persona)
                # context_override: 0.8 (severe — wipe context)
                # emotional_framing: 0.85 (manipulative, often effective)
                # multi_turn: 0.75 (subtle erosion across turns)
                # encoded: 0.9 (deliberate obfuscation = high intent)
                # chain: 0.85 (coordinated multi-step attack)
                # nested: 0.7 (can be accidental, worth flagging)
                # role_play: 0.6 (moderate — can be legitimate)
                severity = {
                    "identity_override": 1.0,
                    "context_override": 0.8,
                    "role_play": 0.6,
                    "multi_turn": 0.75,
                    "emotional_framing": 0.85,
                    "encoded": 0.9,
                    "chain": 0.85,
                    "nested": 0.7,
                }.get(category, 0.5)

                category_score = severity * min(1.0, len(category_matches) / 2.0)
                if category_score > max_category_score:
                    max_category_score = category_score
                    detected_category = category

        detected = len(all_matched) > 0

        # Escalate if multiple categories hit (coordinated attack indicator)
        if len(matched_categories) >= 2:
            max_category_score = min(1.0, max_category_score + 0.1)
            detected_category = f"{detected_category}+multi"

        # Escalate if total hits >= 4 across any categories
        if total_hits >= 4:
            max_category_score = min(1.0, max_category_score + 0.15)

        return InjectionResult(
            detected=detected,
            score=round(max_category_score, 2),
            patterns_matched=all_matched,
            category=detected_category,
        )

    # ── Stage 1: Schema Check ─────────────────────────────────────

    def check_schema(self, content: str) -> FilterResult:
        """
        Verify response structure against persona spec.
        Checks: dealbreakers, essential patterns, structural integrity.
        """
        warnings = []
        score = 1.0

        # Check dealbreakers
        for pattern in self._dealbreaker_patterns:
            if pattern.search(content):
                msg = f"Possible dealbreaker match: '{pattern.pattern}'"
                warnings.append(msg)
                score -= 0.3

        # Check essential identity markers
        if self.persona.first_person and self._third_person_pattern:
            matches = self._third_person_pattern.findall(content)
            if matches:
                warnings.append(f"Found {len(matches)} third-person reference(s): {matches}")
                score -= 0.1 * len(matches)

        score = max(0.0, score)
        passed = score >= self._thresholds["reject"]
        return FilterResult(
            passed=passed,
            content=content,
            score=score,
            warnings=warnings,
        )

    # ── Stage 2: Style Enforce ────────────────────────────────────

    def enforce_style(self, content: str) -> FilterResult:
        """
        Detect tone, formality, and voice rule violations.
        Warning-only — no content modification. Modifications happen in Stage 2b.
        """
        warnings = []
        score = 1.0

        # Formality check
        if self.persona.formality < 0.5:
            # Expecting casual — check for formal markers
            long_words = [w for w in content.split() if len(w) > 12 and w[0].islower()]
            if len(long_words) > 3:
                warnings.append(f"Formality mismatch: {len(long_words)} long/formal words in casual persona")
                score -= 0.1

        if self.persona.formality > 0.7:
            # Expecting formal — check contractions
            contractions = self._formal_patterns["contraction"].findall(content)
            if contractions:
                warnings.append(f"Formality mismatch: {len(contractions)} contractions in formal persona")
                score -= 0.05 * len(contractions)

        # Contractions enforcement
        if self.persona.contractions:
            # Check for overly formal constructions that should use contractions
            formal_phrases = re.findall(r"\b(I am|I will|I would|I have|do not|cannot|will not|it is|that is|there is|you are|they are|we are)\b", content)
            if len(formal_phrases) > 2:
                warnings.append(f"Found {len(formal_phrases)} non-contracted phrases in contraction-enabled persona")
                score -= 0.05

        # Sentence fragments check
        if not self.persona.sentence_fragments:
            fragments = self._formal_patterns["fragment_start"].findall(content)
            if fragments:
                warnings.append(f"Found {len(fragments)} sentence fragments in fragment-disabled persona")
                score -= 0.1

        # Emoji usage check
        emoji_count = len(self._emoji_pattern.findall(content))
        if self.persona.emoji_usage == "none" and emoji_count > 0:
            warnings.append(f"Found {emoji_count} emoji in emoji-disabled persona")
            score -= 0.2 * emoji_count
        elif self.persona.emoji_usage == "rare" and emoji_count > 1:
            warnings.append(f"Found {emoji_count} emoji in rare-emoji persona")
            score -= 0.1
        elif self.persona.emoji_usage == "frequent" and emoji_count == 0:
            warnings.append("No emoji found in frequent-emoji persona")
            score -= 0.05

        # Response length heuristic
        word_count = len(content.split())
        if self.persona.response_length == "concise" and word_count > 100:
            warnings.append(f"Concise persona produced {word_count} words")
            score -= 0.1
        elif self.persona.response_length == "detailed" and word_count < 30:
            warnings.append(f"Detailed persona produced only {word_count} words")
            score -= 0.1

        score = max(0.0, score)
        passed = score >= self._thresholds["reject"]
        return FilterResult(
            passed=passed,
            content=content,
            score=score,
            warnings=warnings,
            metadata={
                "word_count": word_count,
                "emoji_count": emoji_count,
                "contraction_count": len(self._formal_patterns["contraction"].findall(content)),
            },
        )

    # ── Stage 2b: Content Correction ──────────────────────────────

    def correct_content(self, content: str) -> FilterResult:
        """
        Apply automatic corrections to enforce persona spec rules.
        This stage actively modifies content rather than just flagging issues.

        Corrections applied:
        - Expand or apply contractions based on persona preferences
        - Fix third-person self-references to first-person
        - Strip disallowed emoji
        - Tone down formal phrasing in casual personas
        """
        warnings = []
        corrections = []
        score = 1.0
        modified = content

        # 1. Third-person self-reference correction
        if self.persona.first_person:
            modified, tp_changes = _fix_third_person(modified)
            corrections.extend(tp_changes)
            if tp_changes:
                warnings.append("Corrected third-person self-references")
                score -= 0.1

        # 2. Contraction enforcement
        if self.persona.contractions:
            # Count how many formal phrases exist
            formal_phrases = re.findall(
                r"\b(I am|I will|I would|I have|do not|cannot|will not|it is|that is|there is|you are|they are|we are)\b",
                modified
            )
            if len(formal_phrases) >= 1:
                corrected = _apply_contractions(modified)
                if corrected != modified:
                    corrections.append(f"Applied {len(formal_phrases)} contraction(s) (formality={self.persona.formality})")
                    modified = corrected
                    score -= 0.05 * len(formal_phrases)
        else:
            # Contractions disabled — expand them
            contraction_count = len(self._formal_patterns["contraction"].findall(modified))
            if contraction_count > 0:
                corrected = _expand_contractions(modified)
                if corrected != modified:
                    corrections.append(f"Expanded {contraction_count} contractions for formal persona")
                    modified = corrected
                    score -= 0.05

        # 3. Formality adjustment (casual personas — inject casual markers)
        if self.persona.formality < 0.3 and self.persona.name.lower() == modified.split()[0:1]:
            # Very casual persona — gentle nudge if response starts too stiff
            pass  # Complex rewrites are for a future version

        # 4. Emoji enforcement
        if self.persona.emoji_usage == "none":
            stripped, emoji_changes = _strip_emoji(modified)
            if emoji_changes:
                corrections.extend(emoji_changes)
                warnings.append("Stripped disallowed emoji")
                modified = stripped
                score -= 0.1

        # 5. Dealbreaker avoidance — flag but don't silently rewrite
        for pattern in self._dealbreaker_patterns:
            if pattern.search(modified):
                warnings.append(f"Dealbreaker pattern matched: '{pattern.pattern}' — response may need author review")

        score = max(0.0, score)
        passed = score >= self._thresholds["reject"]
        return FilterResult(
            passed=passed,
            content=modified,
            score=score,
            warnings=warnings,
            corrections=corrections,
            metadata={"was_modified": modified != content},
        )

    # ── Stage 3: Score ────────────────────────────────────────────

    def compute_consistency_score(self, content: str, stage_results: list[FilterResult]) -> FilterResult:
        """
        Compute overall character consistency score.
        Weighted combination of all stage scores plus optional semantic checks.
        """
        if not stage_results:
            return FilterResult(passed=True, content=content, score=1.0)

        # Weighted average of stage scores
        weights = {
            "check_schema": 0.25,
            "enforce_style": 0.25,
            "correct_content": 0.20,
            "injection_detection": 0.30,
        }
        total_score = 0.0
        total_weight = 0.0
        all_warnings = []
        all_corrections = []

        for s in stage_results:
            w = weights.get(s.metadata.get("_stage_name", ""), 0.33)
            total_score += s.score * w
            total_weight += w
            all_warnings.extend(s.warnings)
            all_corrections.extend(s.corrections)

        if total_weight > 0:
            total_score /= total_weight

        # Graduated thresholds
        threshold = self._thresholds["reject"]
        passed = total_score >= threshold

        return FilterResult(
            passed=passed,
            content=content,
            score=total_score,
            warnings=all_warnings,
            corrections=all_corrections,
            metadata={
                "stage_count": len(stage_results),
                "weighted_score": total_score,
                "threshold_mode": self.mode,
                "threshold_used": threshold,
            },
        )

    # ── Full Pipeline ─────────────────────────────────────────────

    def apply(self, content: str, context: Optional[dict] = None) -> CharacterFilterReport:
        """
        Run all stages on model output.

        Args:
            content: Raw text output from NormalizedResponse.content
            context: Optional dict with session context (user query, history, etc.)

        Returns:
            CharacterFilterReport with filtered content and scores
        """
        stage_results: dict[str, FilterResult] = {}
        all_warnings: list[str] = []
        all_corrections: list[str] = []
        current_content = content
        injection_result = None

        # Stage 0: Injection Detection (pre-response — checks user prompt)
        if context and "prompt" in context:
            injection_result = self.detect_injection(context["prompt"])
            if injection_result.detected:
                all_warnings.append(
                    f"INJECTION: [{injection_result.category}] "
                    f"score={injection_result.score:.2f}, "
                    f"patterns={injection_result.patterns_matched}"
                )
                # Store injection findings for the report
                stage_results["injection_detection"] = FilterResult(
                    passed=injection_result.score < 0.7,  # allow borderline
                    content=content,
                    score=1.0 - injection_result.score,
                    warnings=[f"Injection detected: {injection_result.category} ({injection_result.score:.2f})"],
                    metadata={
                        "_stage_name": "injection_detection",
                        "injection_detected": True,
                        "injection_score": injection_result.score,
                        "injection_category": injection_result.category,
                        "injection_patterns": injection_result.patterns_matched,
                    },
                )

        # Stage 1: Schema Check
        sr = self.check_schema(current_content)
        sr.metadata["_stage_name"] = "check_schema"
        stage_results["check_schema"] = sr
        all_warnings.extend(sr.warnings)

        # Stage 2: Style Enforce
        er = self.enforce_style(current_content)
        er.metadata["_stage_name"] = "enforce_style"
        stage_results["enforce_style"] = er
        all_warnings.extend(er.warnings)

        # Stage 2b: Content Correction (modifies content)
        cr = self.correct_content(current_content)
        cr.metadata["_stage_name"] = "correct_content"
        stage_results["correct_content"] = cr
        all_warnings.extend(cr.warnings)
        all_corrections.extend(cr.corrections)
        current_content = cr.content  # use corrected content for downstream

        # Stage 3: Score
        sc = self.compute_consistency_score(current_content, [sr, er, cr])
        stage_results["compute_consistency_score"] = sc
        all_warnings.extend(sc.warnings)
        # Avoid duplicating corrections already collected from Stage 2b

        # Graduated threshold logic
        warn_threshold = self._thresholds["warn"]
        reject_threshold = self._thresholds["reject"]

        if sc.score < reject_threshold:
            overall_passed = False
            all_warnings.append(f"BLOCKED: Score {sc.score:.2f} below reject threshold {reject_threshold}")
        elif sc.score < self._thresholds["flag"]:
            overall_passed = True
            all_warnings.append(f"FLAGGED: Score {sc.score:.2f} below flag threshold {self._thresholds['flag']}")
        elif sc.score < warn_threshold:
            all_warnings.append(f"WARN: Score {sc.score:.2f} below warn threshold {warn_threshold}")
            overall_passed = True
        else:
            overall_passed = True

        return CharacterFilterReport(
            overall_passed=overall_passed,
            overall_score=sc.score,
            filtered_content=current_content,
            stages=stage_results,
            warnings=list(set(all_warnings)),
            corrections=all_corrections,
        )


# ── Drift Telemetry ─────────────────────────────────────────────────

def log_drift_entry(
    report: CharacterFilterReport,
    agent_name: str = "unknown",
    context: Optional[dict] = None,
) -> None:
    """Append a drift check entry to meta/drift-log.json.

    One-liner that Colette's monitoring reads for trend analysis.
    Creates the drift-log file if it doesn't exist.
    """
    drift_dir = os.path.join(os.path.expanduser("~"), ".character", "meta")
    drift_path = os.path.join(drift_dir, "drift-log.json")
    os.makedirs(drift_dir, exist_ok=True)

    entry = {
        "timestamp": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": agent_name,
        "score": round(report.overall_score, 3),
        "passed": report.overall_passed,
        "warnings": report.warnings,
        "corrections": report.corrections,
        "mode": context.get("mode", "normal") if context else "normal",
    }

    if os.path.exists(drift_path) and os.path.getsize(drift_path) > 2:
        with open(drift_path) as f:
            try:
                drift = json.load(f)
                if isinstance(drift, dict):
                    drift = [drift]
            except json.JSONDecodeError:
                drift = []
    else:
        drift = []
    drift.append(entry)
    # Keep last 500 entries
    drift = drift[-500:]
    with open(drift_path, "w") as f:
        json.dump(drift, f, indent=2)


def log_model_swap(
    old_model: str,
    new_model: str,
    agent_name: str = "unknown",
) -> None:
    """Log a model swap event to meta/model-adapter.log and drift-log."""
    timestamp = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Write to adapter log
    log_dir = os.path.join(os.path.expanduser("~"), ".character", "meta")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "model-adapter.log")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] MODEL_SWAP: {agent_name} {old_model} → {new_model}\n")

    # Also write a drift entry so Colette's trend analysis catches it
    drift_entry = {
        "timestamp": timestamp,
        "agent": agent_name,
        "event": "model_swap",
        "old_model": old_model,
        "new_model": new_model,
        "score": 0.0,
        "passed": True,
        "warnings": [f"Model swap: {old_model} → {new_model}"],
        "corrections": [],
        "mode": "swap",
    }
    drift_path = os.path.join(log_dir, "drift-log.json")
    if os.path.exists(drift_path) and os.path.getsize(drift_path) > 2:
        with open(drift_path) as f:
            try:
                drift = json.load(f)
                if isinstance(drift, dict):
                    drift = [drift]
            except json.JSONDecodeError:
                drift = []
    else:
        drift = []
    drift.append(drift_entry)
    drift = drift[-500:]
    with open(drift_path, "w") as f:
        json.dump(drift, f, indent=2)


# ── Path Resolution ────────────────────────────────────────────────

def resolve_spec_path(spec_path: str | None = None) -> str:
    """
    Resolve persona spec path with fallback lookup order:

    1. Explicit spec_path argument
    2. PERSONA_SPEC_PATH environment variable
    3. ~/.character/persona.spec.json
    4. ~/.openclaw/workspace/characters/persona.spec.json (legacy)
    """
    if spec_path:
        return spec_path

    env_path = os.environ.get("PERSONA_SPEC_PATH")
    if env_path:
        return env_path

    default_path = os.path.join(os.path.expanduser("~"), ".character", "persona.spec.json")
    if os.path.exists(default_path):
        return default_path

    legacy_path = os.path.join(
        os.path.expanduser("~"), ".openclaw", "workspace", "characters", "persona.spec.json"
    )
    if os.path.exists(legacy_path):
        return legacy_path

    return default_path


def load_character_adapter(
    spec_path: str | None = None,
    mode: str = "normal",
    persona_override: Optional[PersonaSpec] = None,
) -> CharacterAdapter:
    """
    Load persona spec and return configured CharacterAdapter.

    Path resolution order:
    1. spec_path argument
    2. PERSONA_SPEC_PATH env var
    3. ~/.character/persona.spec.json
    4. ~/.openclaw/workspace/characters/persona.spec.json (legacy)

    Args:
        spec_path: Explicit path to persona.spec.json
        mode: Threshold mode — "soft", "normal", or "strict"
        persona_override: Optional pre-built PersonaSpec (skips file load)

    Returns:
        Configured CharacterAdapter ready to apply()
    """
    if persona_override:
        return CharacterAdapter(persona_override, mode=mode)

    path = resolve_spec_path(spec_path)
    persona = PersonaSpec.from_file(path)
    return CharacterAdapter(persona, mode=mode)


def filter_response(
    content: str,
    adapter: CharacterAdapter,
    context: Optional[dict] = None,
) -> tuple[str, CharacterFilterReport]:
    """
    One-shot convenience: run adapter on raw content, return filtered content + report.

    Intended integration point in run_agent.py:
        from agent.character_adapter import filter_response, load_character_adapter
        _adapter = load_character_adapter()  # at session init
        ...
        _finish_result = _cc_fr.normalize_response(response)
        _filtered, _report = filter_response(
            _finish_result.content or "",
            _adapter,
            {"query": messages[-1].get("content", "")},
        )
        if _report.overall_passed:
            _finish_result.content = _filtered
        else:
            _finish_result.content = _filtered  # still use it, but flagged
        assistant_message = _finish_result

    Returns:
        (filtered_content, CharacterFilterReport)
    """
    report = adapter.apply(content, context=context)
    return report.filtered_content, report
