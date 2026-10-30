"""Spoken-language codes the transcription ladder understands.

The table is Whisper's own language list (openai/whisper ``tokenizer.py``,
MIT licence): every code faster-whisper and the Whisper cloud APIs accept,
with the English name the cloud APIs report back in ``verbose_json``.
"""
from __future__ import annotations

from watch_skill.errors import TranscriptionError

WHISPER_LANGUAGES: dict[str, str] = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}

_BY_NAME = {name: code for code, name in WHISPER_LANGUAGES.items()}
# Names and codes people reach for that Whisper spells differently.
_ALIASES = {"burmese": "my", "castilian": "es", "flemish": "nl", "haitian": "ht",
            "letzeburgesch": "lb", "moldavian": "ro", "moldovan": "ro",
            "pushto": "ps", "panjabi": "pa", "valencian": "ca", "jv": "jw",
            "iw": "he", "nb": "no", "mandarin": "zh"}


def normalize_language(value: str | None) -> str | None:
    """A Whisper language code for a code or English name, else None.

    Case-insensitive; a region or script suffix is dropped (``en-US`` and
    ``pt_BR`` become ``en`` and ``pt``), and ``english`` becomes ``en``.
    """
    if value is None:
        return None
    text = value.strip().lower().replace("_", "-")
    if not text:
        return None
    for candidate in (text, text.split("-", 1)[0]):
        if candidate in WHISPER_LANGUAGES:
            return candidate
        code = _BY_NAME.get(candidate) or _ALIASES.get(candidate)
        if code:
            return code
    return None


def validate_language(value: str | None) -> str | None:
    """Normalize a caller's language, or raise before any work starts.

    ``None``, an empty string and ``auto`` all mean "auto-detect" and return
    None. Anything else must name a Whisper language.
    """
    if value is None or value.strip().lower() in {"", "auto"}:
        return None
    code = normalize_language(value)
    if code is None:
        raise TranscriptionError(
            f"unknown spoken language {value!r}",
            code="transcribe.bad_language",
            fix="pass an ISO 639-1 code such as 'en', 'es', 'ar', or omit it to auto-detect",
            details={"language": value},
        )
    return code
