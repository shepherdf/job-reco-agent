import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "data" / "skill_taxonomy.json"


class Taxonomy:
    def __init__(self, path: Path | str = DEFAULT_TAXONOMY_PATH):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.raw = json.load(f)
        self.families: Dict[str, dict] = self.raw["families"]
        self.skills: Dict[str, dict] = self.raw["skills"]
        self.education_levels: Dict[str, int] = self.raw["education_levels"]
        self.cities: List[str] = self.raw["cities"]
        self.metadata: dict = self.raw.get("metadata", {})
        self.skill_names: List[str] = list(self.skills.keys())
        self.family_ids: List[str] = list(self.families.keys())
        self.family_names: Dict[str, str] = {fid: f["name"] for fid, f in self.families.items()}
        self.name_to_family: Dict[str, str] = {v: k for k, v in self.family_names.items()}
        self.alias_to_skill: Dict[str, str] = self._build_alias_map()
        self.regexes: List[tuple[re.Pattern, str]] = self._build_regexes()
        self.skill_family_map: Dict[str, Set[str]] = self._build_skill_family_map()

    def _build_alias_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for canonical, meta in self.skills.items():
            candidates = [canonical] + list(meta.get("aliases", []))
            if canonical == "R":
                candidates = list(meta.get("aliases", []))
            for alias in candidates:
                key = self._normalize_key(alias)
                if not key:
                    continue
                if key not in mapping:
                    mapping[key] = canonical
        return mapping

    def _build_regexes(self) -> List[tuple[re.Pattern, str]]:
        patterns: List[tuple[re.Pattern, str]] = []
        seen: Set[str] = set()
        for alias, canonical in self.alias_to_skill.items():
            if alias in seen:
                continue
            seen.add(alias)
            if re.fullmatch(r"[\x00-\x7f]+", alias):
                escaped = re.escape(alias)
                pattern = re.compile(rf"(?<![A-Za-z0-9_#+.\-]){escaped}(?![A-Za-z0-9_#+.\-])", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
            patterns.append((pattern, canonical))
        return patterns

    def _build_skill_family_map(self) -> Dict[str, Set[str]]:
        result: Dict[str, Set[str]] = {s: set() for s in self.skill_names}
        for fid, fam in self.families.items():
            for skill in fam.get("core_skills", []) + fam.get("optional_skills", []):
                if skill in result:
                    result[skill].add(fid)
        return result

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def normalize_skill(self, value: str) -> str | None:
        key = self._normalize_key(value)
        return self.alias_to_skill.get(key)

    def normalize_skills(self, values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values:
            canonical = self.normalize_skill(value)
            if canonical and canonical not in seen:
                result.append(canonical)
                seen.add(canonical)
        return result

    def extract_skills(self, text: str) -> List[str]:
        if not text:
            return []
        matches: List[tuple[int, int, str]] = []
        for pattern, canonical in self.regexes:
            for m in pattern.finditer(text):
                matches.append((m.start(), -len(m.group(0)), canonical))
        matches.sort()
        result: List[str] = []
        seen: Set[str] = set()
        for _, _, canonical in matches:
            if canonical not in seen:
                result.append(canonical)
                seen.add(canonical)
        return result

    def family_name(self, family_id: str) -> str:
        return self.family_names.get(family_id, family_id)

    def family_id(self, name: str) -> str | None:
        if name in self.families:
            return name
        return self.name_to_family.get(name)

    def education_rank(self, level: str) -> int:
        return int(self.education_levels.get(level, 0))

    def related_families(self, family_id: str) -> Set[str]:
        fam = self.families.get(family_id, {})
        return set(fam.get("related", []))

    def core_skills(self, family_id: str) -> List[str]:
        return [s for s in self.families[family_id].get("core_skills", []) if s in self.skills]

    def optional_skills(self, family_id: str) -> List[str]:
        return [s for s in self.families[family_id].get("optional_skills", []) if s in self.skills]

    def all_family_skills(self, family_id: str) -> List[str]:
        return list(dict.fromkeys(self.core_skills(family_id) + self.optional_skills(family_id)))

    def jieba_words(self) -> List[str]:
        words: Set[str] = set()
        for canonical, meta in self.skills.items():
            if len(canonical) >= 2:
                words.add(canonical)
            for alias in meta.get("aliases", []):
                if len(alias) >= 2 and not re.fullmatch(r"[\x00-\x7f]+", alias):
                    words.add(alias)
        return sorted(words)


_DEFAULT: Taxonomy | None = None


def get_taxonomy() -> Taxonomy:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Taxonomy()
    return _DEFAULT
