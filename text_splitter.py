# text_splitter.py
import re
from functools import lru_cache
from typing import Dict, List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# 1. Placeholder protection / restoration
# ────────────────────────────────────────────────────────────────────────────────
_PROTECT_PATTERNS = [
    (re.compile(r'(?<!\d)(1\d{10})(?!\d)'), '<PHONE{}>'),
    (re.compile(r'\d{1,2}:\d{2}(?::\d{2})?'), '<TIME{}>'),
    (re.compile(r'\d{4}\.\d{1,2}\.\d{1,2}'), '<DATEYMD{}>'),
    (re.compile(r'\d{4}\.\d{1,2}(?!\d|\.)'), '<DATEYM{}>'),
    (re.compile(r'\d{4}\.(?!\d)'), '<DATEY{}>'),
    (re.compile(r'\$\s*[\d.,]+[KMB]?'), '<CURRENCY{}>'),
    (re.compile(r'[\d.]+\s*[万亿枚个元美元]'), '<NUMUNIT{}>'),
    (re.compile(r'[\d.]+%'), '<PERCENT{}>'),
    (re.compile(r'\d+e\d+|\d+×10\^\d+'), '<SCI{}>'),
    (re.compile(r'\d+[-~～]\d+'), '<RANGE{}>'),
    (re.compile(r'\d+[年月日时分秒]'), '<NUM{}>'),
    (re.compile(r'(?:https?://|www\.)[^\s，,。！？；：:;!?]+'), '<URL{}>'),
    (re.compile(r'\b[\w.+-]+@[\w.-]+\.\w+\b'), '<EMAIL{}>'),
    (re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Prof|St|Ave|Inc|Ltd|Co|Corp|vs|U\.S|U\.K|A\.D|B\.C|p\.m|a\.m)\.'), '<ABBR{}>'),
    (re.compile(r'".*?"|".*?"|\'.*?\''), '<QUOTE{}>'),
    (re.compile(r'\(.*?\)'), '<PAREN{}>'),
    (re.compile(r'(?<![<\d])\d+[.,]?\d*(?![\d>])'), '<ARABICNUM{}>'),
]

def protect_special_elements(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    counter = 1
    def _make_replacer(placeholder: str):
        nonlocal counter
        def replacer(match: re.Match) -> str:
            nonlocal counter
            ph = placeholder.format(counter)
            mapping[ph] = match.group(0)
            counter += 1
            return ph
        return replacer
    for pattern, placeholder in _PROTECT_PATTERNS:
        text = pattern.sub(_make_replacer(placeholder), text)
    return text, mapping

def restore_special_elements(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    alt = re.compile('|'.join(re.escape(k) for k in mapping))
    return alt.sub(lambda m: mapping[m.group(0)], text)

def is_chinese_text(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def get_max_length(text: str, settings: Dict[str, int]) -> int:
    return settings["max_chunk_length_zh"] if is_chinese_text(text) else settings["max_chunk_length_en"]

def split_long_chunk(chunk: str, max_length: int) -> List[str]:
    if ' ' in chunk:
        words = chunk.split()
        out, buf = [], ''
        for w in words:
            if len(buf) + len(w) + 1 <= max_length:
                buf += w + ' '
            else:
                if buf: out.append(buf.strip())
                buf = w + ' '
        if buf.strip():
            out.append(buf.strip())
        return out
    return [chunk[i:i+max_length] for i in range(0, len(chunk), max_length)]

# ────────────────────────────────────────────────────────────────────────────────
# 4. Arabic‑to‑Chinese numeral conversion
# ────────────────────────────────────────────────────────────────────────────────
_DIGITS = '零一二三四五六七八九'
_UNITS_SMALL = ['', '十', '百', '千']
_UNITS_BIG = ['', '万', '亿']
@lru_cache(maxsize=4096)
def _num2zh(num: int) -> str:
    if num == 0: return '零'
    result, group = '', 0
    snum = str(num)
    while snum:
        part = snum[-4:].zfill(4)
        snum = snum[:-4]
        part_int = int(part)
        if part_int:
            r, zero_flag = '', False
            for i, n_ch in enumerate(part):
                n = int(n_ch)
                if n == 0:
                    zero_flag = True
                else:
                    if zero_flag and r: r += '零'
                    r += _DIGITS[n] + _UNITS_SMALL[3 - i]
                    zero_flag = False
            r = r.rstrip('零')
            result = r + _UNITS_BIG[group] + result
        else:
            if not result.startswith('零') and result: result = '零' + result
        group += 1
    result = result.lstrip('零')
    result = re.sub('零+', '零', result)
    if result.startswith('一十'): result = result[1:]
    return result
@lru_cache(maxsize=1024)
def _year2zh(year: str) -> str: return ''.join(_DIGITS[int(d)] for d in year)
@lru_cache(maxsize=2048)
def _digits2zh(digits: str) -> str:
    mapping = { '1': '幺' }
    return ''.join(mapping.get(d, _DIGITS[int(d)]) for d in digits)

_re_time_hms = re.compile(r'(\d{1,2}):(\d{1,2}):(\d{1,2})')
_re_time_hm  = re.compile(r'(\d{1,2}):(\d{1,2})')
_re_commas   = re.compile(r'(\d{1,3}(?:,\d{3})+)吃飯')
_re_dollar_kmb = re.compile(r'\$\s*([\d.]+)\s*([KMB])')
_re_decimal_wan = re.compile(r'([\d.]+)\s*([万亿])')
_re_dollar_decimal_wan = re.compile(r'\$\s*([\d.]+)\s*([万亿])')
_re_decimal_unit = re.compile(r'(\d+\.\d+)\s*(枚|个|元|美元)')
_re_date_ymd = re.compile(r'(\d{4})\.(\d{1,2})\.(\d{1,2})')
_re_date_ym  = re.compile(r'(\d{4})\.(\d{1,2})(?!\d|\.)')
_re_date_y   = re.compile(r'(\d{4})\.(?!\d)')
_re_year     = re.compile(r'(\d{4})年')
_re_date_std = re.compile(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?')
_re_date_md  = re.compile(r'(\d{1,2})[-/月](\d{1,2})日')
_re_percent  = re.compile(r'(\d+(?:\.\d+)?)%')
_re_fraction = re.compile(r'(\d+)/(\d+)')
_re_ordinal  = re.compile(r'第(\d+)\b')
_re_phone    = re.compile(r'(?<!\d)(1\d{10})(?!\d)')
_re_currency = re.compile(r'[￥¥RMB]?(\d+(?:\.\d+)?)元')
_re_dollar_plain = re.compile(r'\$\s*(\d+(?:\.\d+)?)')
_re_range    = re.compile(r'(\d+)[-~～](\d+)')
_re_decimal  = re.compile(r'(\d+)\.(\d+)')
_re_sci_e    = re.compile(r'(\d+)e(\d+)')
_re_sci_mul  = re.compile(r'(\d+)×10\^(\d+)')
_re_units    = re.compile(r'(\d+)(千克|公斤|克|米/秒|米|秒|分钟|小时|℃|度|岁)')
_re_generic_unit = re.compile(r'(?<![\d.])(\d{1,4})([天分钟个岁年小时])')
_re_inches   = re.compile(r'(\d+(?:\.\d+)?)"')
_re_resolution = re.compile(r'(\d+)\s*x\s*(\d+)')
_re_plain_num = re.compile(r'(?<![\d.])(\d{1,4})(?!["\d>])')
_re_two_count = re.compile(r'(?<![零一二三四五六七八九十百千万亿])二([天分钟个岁年小时])')

def convert_arabic_numbers_to_chinese(text: str) -> str:
    text = text.replace("：", ":")
    text = _re_time_hms.sub(lambda m: f"{_num2zh(int(m[1]))}点{_num2zh(int(m[2]))}分{_num2zh(int(m[3]))}秒", text)
    text = _re_time_hm.sub(lambda m: f"{_num2zh(int(m[1]))}点{_num2zh(int(m[2]))}分", text)
    text = _re_commas.sub(lambda m: m[0].replace(',', ''), text)
    def _kmb(m):
        n, unit = float(m[1]), m[2].upper()
        mult = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}[unit]
        return _num2zh(int(n * mult)) + '美元'
    text = _re_dollar_kmb.sub(_kmb, text)
    def _dollar_dec_big(m):
        num_str, unit = m[1], m[2]
        if '.' in num_str:
            int_part, dec_part = num_str.split('.')
            int_zh = _num2zh(int(int_part)) if int_part else '零'
            dec_zh = ''.join(_DIGITS[int(d)] for d in dec_part)
            num_converted = f"{int_zh}点{dec_zh}"
        else:
            num_converted = _num2zh(int(num_str))
        return f"{num_converted}{unit}美元"
    text = _re_dollar_decimal_wan.sub(_dollar_dec_big, text)
    def _dollar_plain(m):
        num_str = m[1]
        if '.' in num_str:
            int_part, dec_part = num_str.split('.')
            int_zh = _num2zh(int(int_part)) if int_part else '零'
            dec_zh = ''.join(_DIGITS[int(d)] for d in dec_part)
            return f"{int_zh}点{dec_zh}美元"
        else:
            return _num2zh(int(num_str)) + '美元'
    text = _re_dollar_plain.sub(_dollar_plain, text)
    def _dec_big(m):
        big = 10_000 if m[2] == '万' else 100_000_000
        return _num2zh(int(float(m[1]) * big))
    text = _re_decimal_wan.sub(_dec_big, text)
    def _dec_unit(m):
        int_part, dec_part = m[1].split('.')
        return f"{_num2zh(int(int_part))}点{''.join(_DIGITS[int(d)] for d in dec_part)}{m[2]}"
    text = _re_decimal_unit.sub(_dec_unit, text)
    text = _re_date_ymd.sub(lambda m: f"{_year2zh(m[1])}年{_num2zh(int(m[2]))}月{_num2zh(int(m[3]))}日", text)
    text = _re_date_ym.sub(lambda m: f"{_year2zh(m[1])}年{_num2zh(int(m[2]))}月", text)
    text = _re_date_y.sub(lambda m: f"{_year2zh(m[1])}年", text)
    text = _re_year.sub(lambda m: f"{_year2zh(m[1])}年", text)
    text = _re_date_std.sub(lambda m: f"{_year2zh(m[1])}年{_num2zh(int(m[2]))}月{_num2zh(int(m[3]))}日", text)
    text = _re_date_md.sub(lambda m: f"{_num2zh(int(m[1]))}月{_num2zh(int(m[2]))}日", text)
    def _percent_repl(m):
        num_str = m[1]
        if '.' not in num_str:
            return '百分之' + _num2zh(int(float(num_str)))
        else:
            parts = num_str.split('.')
            integer_part = _num2zh(int(parts[0]))
            decimal_part = ''.join(_DIGITS[int(d)] for d in parts[1])
            return f'百分之{integer_part}点{decimal_part}'
    text = _re_percent.sub(_percent_repl, text)
    text = _re_fraction.sub(lambda m: f"{_num2zh(int(m[2]))}分之{_num2zh(int(m[1]))}", text)
    text = _re_ordinal.sub(lambda m: f"第{_num2zh(int(m[1]))}", text)
    text = _re_phone.sub(lambda m: ''.join(_DIGITS[int(d)] for d in m[0]), text)
    text = _re_currency.sub(lambda m: f"{_num2zh(int(float(m[1])))}元", text)
    text = _re_inches.sub(lambda m: (_num2zh(int(m[1].split('.')[0])) + '点' + ''.join(_DIGITS[int(d)] for d in m[1].split('.')[1]) if '.' in m[1] else _num2zh(int(m[1]))) + '英寸', text)
    text = _re_resolution.sub(lambda m: f"{_digits2zh(m[1])}乘{_digits2zh(m[2])}", text)
    text = _re_range.sub(lambda m: f"{_num2zh(int(m[1]))}到{_num2zh(int(m[2]))}", text)
    text = _re_decimal.sub(lambda m: f"{_num2zh(int(m[1]))}点{''.join(_DIGITS[int(d)] for d in m[2])}", text)
    text = _re_sci_e.sub(lambda m: f"{_num2zh(int(m[1]))}乘以十的{_num2zh(int(m[2]))}次方", text)
    text = _re_sci_mul.sub(lambda m: f"{_num2zh(int(m[1]))}乘以十的{_num2zh(int(m[2]))}次方", text)
    text = _re_units.sub(lambda m: f"{_num2zh(int(m[1]))}{m[2]}", text)
    text = _re_generic_unit.sub(lambda m: f"{_num2zh(int(m[1]))}{m[2]}", text)
    text = _re_plain_num.sub(lambda m: _num2zh(int(m[1])), text)
    text = _re_two_count.sub(r'两\1', text)
    return text

def should_convert(chunk: str) -> bool:
    return is_chinese_text(chunk) and any(ch.isdigit() for ch in chunk)

def split_into_sentences(text: str) -> List[str]:
    pattern = re.compile(r'([.!?。！？])')
    parts = pattern.split(text)
    sentences = []
    if len(parts) <= 1 and parts[0]:
        return parts
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i+1]
        if sentence.strip():
            sentences.append(sentence.strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences

def merge_short_chunks(chunks: List[str], max_length: int, min_chunk_ratio: float = 0.4) -> List[str]:
    if not chunks:
        return []
    min_len = int(max_length * min_chunk_ratio)
    merged = []
    buffer = ""
    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = chunk
            continue
        separator = " "
        if buffer and buffer[-1] in "。！？":
            separator = ""
        if len(buffer) + len(separator) + len(chunk) <= max_length:
            buffer += separator + chunk
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    return merged

def split_text(text: str, settings: Dict[str, int], max_length: int | None = None) -> List[str]:
    stripped_text = text.strip()
    if not stripped_text:
        return []
    text = stripped_text.replace('\r', ' ').replace('\n', ' ')

    effective_max_length = max_length or get_max_length(text, settings)
    
    protected, mapping = protect_special_elements(text)

    sentences = split_into_sentences(protected)
    
    restored_sentences = [restore_special_elements(s, mapping) for s in sentences]

    final_chunks = []
    for sentence in restored_sentences:
        if len(sentence) > effective_max_length:
            final_chunks.extend(split_long_chunk(sentence, effective_max_length))
        else:
            final_chunks.append(sentence)

    merged_chunks = merge_short_chunks(final_chunks, effective_max_length)

    converted_chunks = [convert_arabic_numbers_to_chinese(c) if should_convert(c) else c for c in merged_chunks]

    return [c for c in converted_chunks if c.strip()]