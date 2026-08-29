#!/usr/bin/env python3
"""
Per-net uniform-width guide generator for KiCad boards (v9 and legacy).

For each board folder:
  - Read  {folder}/{stem}.kicad_pcb + {folder}/{stem}.kicad_pro
  - Write {folder}/{stem}{suffix}.kicad_pcb + {folder}/{stem}{suffix}.kicad_pro
           {folder}/{stem}{suffix}_unrouted.kicad_pcb (routing removed)

Algorithm:
  1. Parse net classes from .kicad_pro (v9) or .kicad_pcb net_class blocks (legacy)
  2. Preserve original DRC rules/severities from .kicad_pro verbatim.
     (only fill fields that don't exist with a 0 default.)
  3. Extract per-net segment width (min, max) from segments.
  4. Baseline DRC on original → violation counts by type.
  5. Trial loop (N_TRIAL, linear max→min per-net):
        - Uniform per-net widths → rewrite segments
        - Build new guide .kicad_pcb + .kicad_pro
        - Run DRC → accept if no NEW violations vs baseline
  6. Save first accepting trial (else min-trial).
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from setup_drvzero import remove_routing as legacy_remove_routing
except Exception:
    legacy_remove_routing = None


N_TRIAL = 5
MM2DSN = 1000

# .kicad_pcb 'setup' legacy DRC field -> modern pro rules key
LEGACY_SETUP_TO_RULES = {
    'trace_min': 'min_track_width',
    'trace_clearance': 'min_clearance',
    'via_min_size': 'min_via_diameter',
    'via_min_drill': 'min_through_hole_diameter',
    'uvia_min_size': 'min_microvia_diameter',
    'uvia_min_drill': 'min_microvia_drill',
}

# KiCad pro net_settings.classes parameter name -> internal representation
PRO_CLASS_KEYS = {
    'clearance': 'clearance',
    'track_width': 'track_width',
    'via_diameter': 'via_dia',
    'via_drill': 'via_drill',
    'microvia_diameter': 'uvia_dia',
    'microvia_drill': 'uvia_drill',
}
INTERNAL_TO_PRO = {v: k for k, v in PRO_CLASS_KEYS.items()}

REFILL_HELPER_SRC = '''
import sys, pcbnew
src, dst = sys.argv[1], sys.argv[2]
board = pcbnew.LoadBoard(src)
filler = pcbnew.ZONE_FILLER(board)
filler.Fill(list(board.Zones()), False, None)
board.Save(dst)
'''


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def _balanced_block_end(content, start):
    """Exclusive end index of the block from '(' to its matching ')'."""
    depth = 0
    for j in range(start, len(content)):
        c = content[j]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return j + 1
    return len(content)


def extract_net_id_to_name(content):
    """Top-level (net id name) declarations of a kicad_pcb -> {id: name}."""
    result = {}
    for m in re.finditer(
            r'^\s*\(net\s+(\d+)\s+("(?:[^"\\]|\\.)*"|[^)"]+)\)',
            content, re.MULTILINE):
        nid = m.group(1)
        raw = m.group(2).strip('"').strip()
        result[nid] = raw
    return result


def extract_net_width_range(content):
    """Per-net (min, max) width over the (segment ...) and (arc ...) blocks
    (multi-line blocks included)."""
    per_net = defaultdict(list)
    for tag in ('(segment', '(arc'):
        idx = 0
        while True:
            start = content.find(tag, idx)
            if start == -1:
                break
            end = _balanced_block_end(content, start)
            block = content[start:end]
            wm = re.search(r'\(width\s+([0-9.]+)\)', block)
            nm = re.search(r'\(net\s+(\d+)\)', block)
            if wm and nm:
                nid = nm.group(1)
                if nid != '0':
                    per_net[nid].append(float(wm.group(1)))
            idx = end
    return {nid: (min(ws), max(ws)) for nid, ws in per_net.items()}


def extract_uuid_to_net(content):
    """Track item (segment/arc/via) UUID -> net id."""
    out = {}
    for tag in ('(segment', '(arc', '(via'):
        idx = 0
        while True:
            start = content.find(tag, idx)
            if start == -1:
                break
            end = _balanced_block_end(content, start)
            block = content[start:end]
            um = re.search(r'\(uuid\s+"([^"]+)"\s*\)', block)
            if not um:
                um = re.search(r'\(uuid\s+([0-9a-fA-F\-]+)\s*\)', block)
            nm = re.search(r'\(net\s+(\d+)\)', block)
            if um and nm:
                out[um.group(1)] = nm.group(1)
            idx = end
    return out


def extract_offending_net_ids(violations, uuid_to_net, net_name_to_id):
    """Set of net ids referenced by the DRC violation items.

    1) match on the item UUID first
    2) fall back to the [net_name] pattern in the item description.
    """
    out = set()
    for v in violations or []:
        for it in v.get('items', []) or []:
            uid = it.get('uuid', '') or ''
            nid = uuid_to_net.get(uid)
            if nid is not None:
                out.add(nid)
                continue
            desc = it.get('description', '') or ''
            m = re.search(r'\[([^\]]+)\]', desc)
            if m and m.group(1) in net_name_to_id:
                out.add(net_name_to_id[m.group(1)])
    return out


def extract_setup_legacy_rules(content):
    """Map the legacy DRC keys of the .kicad_pcb (setup ...) block to modern
    rule keys."""
    start = content.find('(setup')
    if start == -1:
        return {}
    end = _balanced_block_end(content, start)
    block = content[start:end]
    rules = {}
    for legacy, modern in LEGACY_SETUP_TO_RULES.items():
        m = re.search(rf'\({legacy}\s+([0-9.]+)\)', block)
        if m:
            rules[modern] = float(m.group(1))
    return rules


def extract_pcb_net_class_blocks(content):
    """Parse the net_class blocks of a legacy (v5/v6) kicad_pcb.
    Returns (class_params, net_to_class) with internal keys.
    """
    class_params = {}
    net_to_class = {}
    idx = 0
    while True:
        start = content.find('(net_class', idx)
        if start == -1:
            break
        end = _balanced_block_end(content, start)
        block = content[start:end]

        m = re.match(r'\(net_class\s+("([^"]+)"|(\S+))', block)
        if not m:
            idx = end
            continue
        cname = (m.group(2) or m.group(3)).strip('"')

        # internal key names (via_dia/via_drill/uvia_*)
        params = {}
        for key in ('clearance', 'trace_width', 'via_dia',
                    'via_drill', 'uvia_dia', 'uvia_drill'):
            pm = re.search(rf'\({key}\s+([0-9.]+)\)', block)
            if pm:
                params[key] = float(pm.group(1))
        class_params[cname] = params

        for mn in re.finditer(
                r'\(add_net\s+("([^"]+)"|([^)"]+))\)', block):
            name = (mn.group(2) or mn.group(3)).strip()
            net_to_class[name] = cname

        idx = end
    return class_params, net_to_class


def extract_pro_net_classes(pro_data):
    """Parse net_settings.classes + netclass_patterns of a v9 .kicad_pro.
    Returns (class_params, explicit_net_to_class, patterns) with internal keys.
    patterns = [(pattern_str, classname), ...] in original order.
    """
    class_params = {}
    net_to_class = {}
    classes = pro_data.get('net_settings', {}).get('classes', []) or []
    for c in classes:
        name = c.get('name')
        if not name:
            continue
        params = {}
        for pro_k, internal_k in PRO_CLASS_KEYS.items():
            if pro_k in c and c[pro_k] is not None:
                params[internal_k] = float(c[pro_k])
        # unify the internal representation: track_width -> trace_width
        if 'track_width' in params:
            params['trace_width'] = params.pop('track_width')
        class_params[name] = params
        for net_name in c.get('nets', []) or []:
            if net_name:
                net_to_class[net_name] = name

    # netclass_patterns: normally where the net -> class mapping is defined.
    patterns = []
    raw_pats = (pro_data.get('net_settings', {})
                        .get('netclass_patterns', []) or [])
    for p in raw_pats:
        if not isinstance(p, dict):
            continue
        pat = p.get('pattern')
        cls = p.get('netclass')
        if pat and cls:
            patterns.append((pat, cls))
    return class_params, net_to_class, patterns


def _kicad_pattern_match(pattern, name):
    """KiCad netclass pattern matching.

    Patterns support *, ? and character classes [..], and are case-sensitive.
    fnmatchcase behaves the same as KiCad's EDA_PATTERN_MATCH_WILDCARD.
    """
    return fnmatch.fnmatchcase(name, pattern)


def derive_net_to_class(net_id_to_name, explicit, patterns, class_params):
    """net -> class mapping.

    Priority: explicit `nets` list > netclass_patterns (first match) > Default.
    """
    out = {}
    valid_patterns = [(p, c) for (p, c) in patterns if c in class_params]
    for name in net_id_to_name.values():
        if not name:
            continue
        if name in explicit:
            out[name] = explicit[name]
            continue
        matched = None
        for pat, cls in valid_patterns:
            if _kicad_pattern_match(pat, name):
                matched = cls
                break
        if matched is not None:
            out[name] = matched
        else:
            out[name] = 'Default' if 'Default' in class_params else None
    return out


# ─────────────────────────────────────────────
# Segment width rewrite (multi-line & single-line)
# ─────────────────────────────────────────────

def apply_net_widths_to_tracks(content, net_id_to_width):
    """Replace the width of every (segment ...) and (arc ...) block with the
    per-net value."""
    def _rewrite(src, tag):
        out = []
        idx = 0
        while idx < len(src):
            start = src.find(tag, idx)
            if start == -1:
                out.append(src[idx:])
                break
            out.append(src[idx:start])
            end = _balanced_block_end(src, start)
            block = src[start:end]
            nm = re.search(r'\(net\s+(\d+)\)', block)
            if nm and nm.group(1) in net_id_to_width:
                new_w = net_id_to_width[nm.group(1)]
                w_str = f'{new_w:.5f}'.rstrip('0').rstrip('.')
                block = re.sub(r'\(width\s+[0-9.]+\)',
                               f'(width {w_str})', block, count=1)
            out.append(block)
            idx = end
        return ''.join(out)

    content = _rewrite(content, '(segment')
    content = _rewrite(content, '(arc')
    return content


def remove_routing_v9(content):
    """Drop the segment, arc and via blocks (multi-line included)."""
    for tag in ('(segment', '(arc', '(via'):
        out = []
        idx = 0
        while idx < len(content):
            start = content.find(tag, idx)
            if start == -1:
                out.append(content[idx:])
                break
            out.append(content[idx:start])
            end = _balanced_block_end(content, start)
            # drop the preceding whitespace/newline as well
            while out and out[-1].endswith(('\n\t', '\n ', '\n')):
                tail = out[-1]
                # trim only the trailing whitespace after the last newline
                m = re.search(r'[\t ]*\n[\t ]*$', tail)
                if m:
                    out[-1] = tail[:m.start()] + '\n'
                break
            idx = end
        content = ''.join(out)
    return content


# ─────────────────────────────────────────────
# Class grouping
# ─────────────────────────────────────────────

def build_groups(net_id_to_name, net_widths, net_to_class,
                 class_params, default_w_fallback,
                 preserved_keys):
    """Group by the (width, params_tuple) key, keeping the original
    parameters verbatim."""
    groups = defaultdict(list)
    for nid, name in net_id_to_name.items():
        if nid == '0' or not name:
            continue
        w = net_widths.get(nid, default_w_fallback)
        cls = net_to_class.get(name) or 'Default'
        src = class_params.get(cls) or class_params.get('Default') or {}
        # parameters to preserve (taken from the class each net belongs to)
        p_items = []
        for k in preserved_keys:
            v = src.get(k)
            if v is not None:
                p_items.append((k, v))
        key = (round(w, 6), tuple(sorted(p_items)))
        groups[key].append(name)
    return dict(groups)


def assign_class_names(groups, default_w):
    """(width, params) → (class_name, params_dict) + original default_w grouping."""
    by_width = defaultdict(list)
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        by_width[key[0]].append(key)

    used = set()
    out = {}
    for w in sorted(by_width):
        for key in by_width[w]:
            params = dict(key[1])
            if w == default_w:
                base = 'Default'
            else:
                s = f'{w:.4f}'.rstrip('0').rstrip('.')
                base = f'guide_W{s}'
            cname, suf = base, 2
            while cname in used:
                cname = f'{base}_v{suf}'
                suf += 1
            used.add(cname)
            out[key] = (cname, params)
    return out


# ─────────────────────────────────────────────
# Output build: .kicad_pro
# ─────────────────────────────────────────────

def build_guide_pro(orig_pro, groups, class_info):
    """Clone the original pro, then replace net_settings.classes."""
    pro = json.loads(json.dumps(orig_pro))  # deep copy
    ns = pro.setdefault('net_settings', {})
    orig_classes = ns.get('classes', []) or []
    orig_by_name = {c['name']: c for c in orig_classes if isinstance(c, dict) and 'name' in c}
    default_template = orig_by_name.get('Default', {})

    # reverse map net_name -> guide class name (used to rewrite the patterns)
    net_to_guide_class = {}
    new_classes = []
    for key in sorted(groups, key=lambda k: (k[0], class_info[k][0])):
        cname, params = class_info[key]
        for net_name in groups[key]:
            net_to_guide_class[net_name] = cname
        w = key[0]
        # base template: the same-named original class if any, else Default
        template = orig_by_name.get(cname) or default_template
        new_cls = json.loads(json.dumps(template)) if template else {}
        new_cls['name'] = cname
        new_cls['track_width'] = w
        # overwrite the preserved parameters (internal -> pro keys)
        for internal_k, val in params.items():
            pro_k = INTERNAL_TO_PRO.get(internal_k)
            if pro_k:
                new_cls[pro_k] = val
        # nets list (omitted for Default — KiCad convention)
        if cname == 'Default':
            new_cls.pop('nets', None)
        else:
            new_cls['nets'] = sorted(groups[key])
        new_classes.append(new_cls)

    ns['classes'] = new_classes

    # netclass_patterns: repoint the original patterns at the guide class
    # names. A pattern equal to an exact net name is replaced by that net's
    # guide class; wildcard patterns fall back to Default (freerouting writes
    # exact net names).
    orig_patterns = ns.get('netclass_patterns', None)
    if orig_patterns is not None:
        new_patterns = []
        for p in orig_patterns:
            pattern = p.get('pattern', '')
            guide_cname = net_to_guide_class.get(pattern, 'Default')
            new_patterns.append({'pattern': pattern, 'netclass': guide_cname})
        ns['netclass_patterns'] = new_patterns

    return pro


# ─────────────────────────────────────────────
# Output build: .kicad_pcb (legacy net_class blocks rewritten)
# ─────────────────────────────────────────────

def _format_net_name_kicad(name):
    if re.search(r'[() /+\-]', name):
        return f'"{name}"'
    return name


def _format_net_class_block(class_name, desc, trace_width, params, net_names):
    lines = [f'  (net_class {class_name} "{desc}"',
             f'    (clearance {params["clearance"]})',
             f'    (trace_width {trace_width})',
             f'    (via_dia {params["via_dia"]})',
             f'    (via_drill {params["via_drill"]})']
    if 'uvia_dia' in params:
        lines.append(f'    (uvia_dia {params["uvia_dia"]})')
    if 'uvia_drill' in params:
        lines.append(f'    (uvia_drill {params["uvia_drill"]})')
    for name in sorted(net_names):
        lines.append(f'    (add_net {_format_net_name_kicad(name)})')
    lines.append('  )')
    return '\n'.join(lines)


def _remove_net_class_blocks(content):
    result = []
    depth = 0
    in_block = False
    for line in content.split('\n'):
        stripped = line.strip()
        if not in_block:
            if re.match(r'\(net_class\b', stripped):
                in_block = True
                depth = stripped.count('(') - stripped.count(')')
                if depth <= 0:
                    in_block = False
            else:
                result.append(line)
        else:
            depth += stripped.count('(') - stripped.count(')')
            if depth <= 0:
                in_block = False
    return '\n'.join(result)


def rebuild_pcb_net_class_blocks(content, groups, class_info):
    """Rewrite the net_class blocks of a legacy kicad_pcb."""
    content_no = _remove_net_class_blocks(content)

    blocks = []
    for key in sorted(groups, key=lambda k: (k[0], class_info[k][0])):
        cname, params = class_info[key]
        desc = 'This is the default net class.' if cname == 'Default' else ''
        # legacy pcb blocks require via_dia/via_drill
        if 'via_dia' not in params or 'via_drill' not in params:
            continue
        blocks.append(_format_net_class_block(
            cname, desc, key[0], params, groups[key]))
    if not blocks:
        return content_no
    new_text = '\n\n'.join(blocks) + '\n'

    m = re.search(r'^\s*\((?:module|footprint)\b',
                  content_no, re.MULTILINE)
    if m:
        return content_no[:m.start()] + new_text + '\n' + content_no[m.start():]
    return content_no + '\n' + new_text


# ─────────────────────────────────────────────
# DRC run
# ─────────────────────────────────────────────

def _write_refill_helper():
    helper = Path('/tmp/_make_guide_refill_helper.py')
    if not helper.exists() or helper.read_text() != REFILL_HELPER_SRC:
        helper.write_text(REFILL_HELPER_SRC)
    return helper


def _refill_zones(pcb_src, pro_src, dst_pcb, dst_pro, helper):
    """Zone refill through the pcbnew API; the pro needs the same stem."""
    try:
        shutil.copy(pro_src, dst_pro)
    except Exception:
        pass
    try:
        subprocess.run(
            ['/usr/bin/python3', str(helper), str(pcb_src), str(dst_pcb)],
            capture_output=True, timeout=180, check=False)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return dst_pcb.exists()


def run_drc(pcb_content, pro_data, tmpdir, tag='drc'):
    """
    Write pcb/pro into tmpdir, refill zones, then run kicad-cli drc.
    Returns dict {'counts': {type: count}, 'violations': [...]}
    or None on failure.
    """
    tmp = Path(tmpdir)
    pcb_src = tmp / f'{tag}.kicad_pcb'
    pro_src = tmp / f'{tag}.kicad_pro'
    pcb_src.write_text(pcb_content, encoding='utf-8')
    pro_src.write_text(json.dumps(pro_data), encoding='utf-8')

    helper = _write_refill_helper()
    refilled_pcb = tmp / f'{tag}_refilled.kicad_pcb'
    refilled_pro = tmp / f'{tag}_refilled.kicad_pro'
    if _refill_zones(pcb_src, pro_src, refilled_pcb, refilled_pro, helper):
        target = refilled_pcb
    else:
        target = pcb_src

    out = tmp / f'{tag}.json'
    try:
        subprocess.run(
            ['kicad-cli', 'pcb', 'drc', '--format', 'json',
             '--severity-error', '--output', str(out), str(target)],
            capture_output=True, timeout=180, check=False)
    except Exception:
        return None
    if not out.exists():
        return None
    try:
        data = json.loads(out.read_text())
    except Exception:
        return None
    violations = data.get('violations', []) or []
    counts = defaultdict(int)
    for v in violations:
        counts[v.get('type', 'unknown')] += 1
    return {'counts': dict(counts), 'violations': violations}


def drc_passes(counts):
    """Pass when there are zero violations. The inputs already pass DRC, so no
    baseline diff is needed."""
    return sum(counts.values()) == 0


# ─────────────────────────────────────────────
# Board processing
# ─────────────────────────────────────────────

def _detect_default_width(class_params, observed_widths):
    """Default width, used for nets without segments and for class naming.

    Priority: Default class trace_width -> smallest observed width -> 0.25
    """
    default = class_params.get('Default', {})
    tw = default.get('trace_width')
    if tw and tw > 0:
        return tw
    if observed_widths:
        return min(observed_widths)
    return 0.25


def _compute_widths_per_t(t_per_net, net_wrange, default_w, min_track_w=0.0):
    """Width per net from its t (0 = max, 1 = min).

    A net with no segments is pinned to default_w regardless of t.
    The lower floor is the wider of the net's own observed minimum and the
    board's min_track_width rule; a net class's trace_width is deliberately
    not a floor, so the widths the boards ship with are reproducible.
    """
    widths = {}
    for nid, t in t_per_net.items():
        if nid in net_wrange:
            mn, mx = net_wrange[nid]
            mn_floor = max(mn, min_track_w)
            widths[nid] = mx * (1 - t) + mn_floor * t
        else:
            widths[nid] = default_w
    return widths


def process_board(folder, input_stem='processed_v9', guide_suffix='_guide',
                  verbose=False):
    """
    Returns dict:
      {'folder': str, 'status': 'pass'|'fail'|'skip'|'error',
       'trial': int, 'baseline': dict, 'final': dict, 'msg': str}
    """
    folder = Path(folder)
    pcb_path = folder / f'{input_stem}.kicad_pcb'
    pro_path = folder / f'{input_stem}.kicad_pro'
    guide_pcb = folder / f'{input_stem}{guide_suffix}.kicad_pcb'
    guide_pro = folder / f'{input_stem}{guide_suffix}.kicad_pro'
    guide_unrouted_pcb = folder / f'{input_stem}{guide_suffix}_unrouted.kicad_pcb'

    res = {'folder': folder.name, 'status': 'error',
           'trial': 0, 'final': {}, 'msg': ''}

    try:
        if not pcb_path.exists():
            res['status'] = 'skip'
            res['msg'] = 'no pcb'
            return res
        content = pcb_path.read_text(encoding='utf-8')

        pro_data = {}
        if pro_path.exists():
            try:
                pro_data = json.loads(pro_path.read_text())
            except Exception:
                pro_data = {}

        # 1) parse
        net_id_to_name = extract_net_id_to_name(content)
        pcb_classes, pcb_n2c = extract_pcb_net_class_blocks(content)
        pro_classes, pro_n2c, pro_patterns = extract_pro_net_classes(pro_data)

        # v9: no pcb net_class -> prefer pro / legacy: prefer pcb
        is_v9 = (len(pcb_classes) == 0 and len(pro_classes) > 0)
        if is_v9:
            class_params = pro_classes
            explicit_n2c = pro_n2c
            patterns = pro_patterns
        else:
            class_params = pcb_classes or pro_classes
            explicit_n2c = pcb_n2c or pro_n2c
            # legacy pcb has no pattern mechanism; the pro patterns are used
            # only when no pcb classes were found.
            patterns = [] if pcb_classes else pro_patterns

        if 'Default' not in class_params:
            res['status'] = 'skip'
            res['msg'] = 'no Default class'
            return res

        net_to_class = derive_net_to_class(
            net_id_to_name, explicit_n2c, patterns, class_params)
        net_wrange = extract_net_width_range(content)

        if not net_wrange:
            res['status'] = 'skip'
            res['msg'] = 'no routed segments'
            return res

        observed_ws = [w for r in net_wrange.values() for w in r]
        default_w = _detect_default_width(class_params, observed_ws)

        # min_track_width rule — the lower floor of _compute_widths_per_t
        _rules = (pro_data.get('board', {})
                  .get('design_settings', {}).get('rules', {}))
        _legacy_rules = extract_setup_legacy_rules(content)
        min_track_w = float(
            _rules.get('min_track_width')
            or _legacy_rules.get('min_track_width')
            or 0.0
        )

        # parameter keys to preserve (only the ones present)
        preserved_keys_all = ['clearance', 'via_dia', 'via_drill',
                              'uvia_dia', 'uvia_drill']
        preserved_keys = [k for k in preserved_keys_all
                          if any(k in v for v in class_params.values())]

        # helper maps used to identify the offending nets
        uuid_to_net = extract_uuid_to_net(content)
        net_name_to_id = {n: i for i, n in net_id_to_name.items()
                          if n and i != '0'}

        # every net starts at t = 0 (= max width).
        t_per_net = {nid: 0.0 for nid in net_id_to_name
                     if nid != '0' and net_id_to_name[nid]}
        step = 1.0 / max(1, N_TRIAL - 1)

        with tempfile.TemporaryDirectory(prefix='make_guide_') as tmp:
            # Trial loop — each trial raises t by one step (= narrows the
            # width) only for the nets that violated DRC in the previous
            # trial; every other net keeps its max width.
            accepted = None
            last = None
            offender_history = []  # debug trace
            for trial in range(N_TRIAL):
                net_widths = _compute_widths_per_t(
                    t_per_net, net_wrange, default_w, min_track_w)
                modified_pcb = apply_net_widths_to_tracks(
                    content, net_widths)

                groups = build_groups(
                    net_id_to_name, net_widths, net_to_class,
                    class_params, default_w, preserved_keys)
                observed_group_ws = sorted({k[0] for k in groups})
                trial_default_w = (observed_group_ws[0] if observed_group_ws
                                   else default_w)
                class_info = assign_class_names(groups, trial_default_w)

                # guide pro (carrying the new classes)
                trial_pro = build_guide_pro(pro_data, groups, class_info)

                # legacy boards need the pcb blocks rewritten too
                if not is_v9:
                    modified_pcb = rebuild_pcb_net_class_blocks(
                        modified_pcb, groups, class_info)

                drc_result = run_drc(modified_pcb, trial_pro, tmp,
                                     tag=f'trial{trial}')
                if drc_result is None:
                    if verbose:
                        print(f'  [{folder.name}] trial {trial+1}: DRC error')
                    continue

                trial_counts = drc_result['counts']
                last = dict(pcb=modified_pcb, pro=trial_pro,
                            groups=groups, class_info=class_info,
                            counts=trial_counts,
                            t_snapshot=dict(t_per_net))

                if drc_passes(trial_counts):
                    accepted = last
                    accepted['trial'] = trial + 1
                    break

                # identify the violating nets -> narrow only those by a step.
                offender_ids = extract_offending_net_ids(
                    drc_result['violations'], uuid_to_net, net_name_to_id)
                # reducible: nets with t < 1 present in net_wrange (segments).
                targets = [nid for nid in offender_ids
                           if nid in t_per_net
                           and nid in net_wrange
                           and t_per_net[nid] < 1.0]
                if not targets:
                    # nothing identified, or all at min -> narrow all (fallback).
                    targets = [nid for nid in t_per_net
                               if nid in net_wrange
                               and t_per_net[nid] < 1.0]
                offender_history.append(
                    {'trial': trial, 'offenders': len(offender_ids),
                     'reduced': len(targets)})
                for nid in targets:
                    t_per_net[nid] = min(1.0, t_per_net[nid] + step)

            if accepted is None:
                # Final fallback: force every reducible net to its min width.
                # When the trial loop's step accumulation leaves nets at t < 1,
                # the original segments' min width is known routable, so one
                # more attempt is made there.
                final_t = {nid: (1.0 if nid in net_wrange else t)
                           for nid, t in t_per_net.items()}
                final_widths = _compute_widths_per_t(
                    final_t, net_wrange, default_w, min_track_w)
                final_pcb = apply_net_widths_to_tracks(
                    content, final_widths)
                final_groups = build_groups(
                    net_id_to_name, final_widths, net_to_class,
                    class_params, default_w, preserved_keys)
                final_obs_ws = sorted({k[0] for k in final_groups})
                final_default_w = (final_obs_ws[0] if final_obs_ws
                                   else default_w)
                final_class_info = assign_class_names(
                    final_groups, final_default_w)
                final_pro = build_guide_pro(
                    pro_data, final_groups, final_class_info)
                if not is_v9:
                    final_pcb = rebuild_pcb_net_class_blocks(
                        final_pcb, final_groups, final_class_info)
                final_drc = run_drc(final_pcb, final_pro, tmp,
                                    tag='final_min')
                final_accepted = None
                if final_drc is not None:
                    final_counts = final_drc['counts']
                    final_record = dict(
                        pcb=final_pcb, pro=final_pro,
                        groups=final_groups, class_info=final_class_info,
                        counts=final_counts, t_snapshot=dict(final_t))
                    if drc_passes(final_counts):
                        final_accepted = final_record
                        final_accepted['trial'] = N_TRIAL + 1
                    else:
                        # both fail -> keep whichever has fewer violations.
                        last_v = (sum(last['counts'].values())
                                  if last else float('inf'))
                        final_v = sum(final_counts.values())
                        if last is None or final_v < last_v:
                            accepted = final_record
                            accepted['trial'] = N_TRIAL + 1
                        else:
                            accepted = last
                            accepted['trial'] = N_TRIAL
                if final_accepted is not None:
                    accepted = final_accepted
                elif accepted is None:
                    if last is None:
                        res['status'] = 'error'
                        res['msg'] = 'all DRC trials errored'
                        return res
                    accepted = last
                    accepted['trial'] = N_TRIAL

            # 4) save
            guide_pcb.write_text(accepted['pcb'], encoding='utf-8')
            guide_pro.write_text(json.dumps(accepted['pro'], indent=2),
                                 encoding='utf-8')
            unrouted = remove_routing_v9(accepted['pcb'])
            guide_unrouted_pcb.write_text(unrouted, encoding='utf-8')

            # stats: nets left at max (t=0) / nets narrowed
            t_snap = accepted.get('t_snapshot', {})
            n_at_max = sum(1 for v in t_snap.values() if v == 0.0)
            n_reduced = sum(1 for v in t_snap.values() if v > 0.0)
            n_at_min = sum(1 for v in t_snap.values() if v >= 1.0)

            res['status'] = ('pass' if drc_passes(accepted['counts'])
                             else 'fail')
            res['trial'] = accepted['trial']
            res['final'] = accepted['counts']
            res['n_nets_at_max'] = n_at_max
            res['n_nets_reduced'] = n_reduced
            res['n_nets_at_min'] = n_at_min
            res['offender_history'] = offender_history
            return res

    except Exception as e:
        res['status'] = 'error'
        res['msg'] = f'{type(e).__name__}: {e}'
        if verbose:
            traceback.print_exc()
        return res


# ─────────────────────────────────────────────
# Main / CLI
# ─────────────────────────────────────────────

def _process_wrapper(args):
    folder, stem, suffix = args
    return process_board(folder, stem, suffix, verbose=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', required=True,
                   help='board-folder root to process (e.g. the drc_fix_v9.py '
                        'output directory)')
    p.add_argument('--stem', default='processed_v9')
    p.add_argument('--suffix', default='_guide')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0,
                   help='0=all, N=only first N boards')
    p.add_argument('--targets', nargs='*', default=None,
                   help='specific folder names')
    p.add_argument('--log', default='/tmp/make_guide_log.json')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    base = Path(args.base_dir)
    if args.targets:
        folders = [base / t for t in args.targets if (base / t).is_dir()]
    else:
        folders = sorted(f for f in base.iterdir() if f.is_dir())
    if args.limit:
        folders = folders[:args.limit]

    print(f'Processing {len(folders)} boards under {base}')
    print(f'  stem={args.stem}  suffix={args.suffix}  workers={args.workers}')
    print(f'  log={args.log}')

    t0 = time.time()
    results = []
    if args.workers <= 1:
        for f in folders:
            r = process_board(f, args.stem, args.suffix,
                              verbose=args.verbose)
            results.append(r)
            if args.verbose:
                print(f"  {r['status']:5s} {r['folder']}  "
                      f"trial={r['trial']}  msg={r['msg']}")
    else:
        work = [(str(f), args.stem, args.suffix) for f in folders]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_wrapper, w): w for w in work}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    w = futs[fut]
                    r = {'folder': Path(w[0]).name,
                         'status': 'error', 'trial': 0,
                         'baseline': {}, 'final': {},
                         'msg': f'{type(e).__name__}: {e}'}
                results.append(r)
                done += 1
                if done % 25 == 0:
                    elapsed = time.time() - t0
                    print(f'  {done}/{len(folders)}  '
                          f'{elapsed:.0f}s elapsed')

    Path(args.log).write_text(json.dumps(results, indent=2),
                              encoding='utf-8')

    counts = defaultdict(int)
    for r in results:
        counts[r['status']] += 1
    print(f'\n=== Summary ({time.time()-t0:.0f}s) ===')
    for k, v in sorted(counts.items()):
        print(f'  {k:6s}: {v}')
    print(f'\nDetailed log: {args.log}')


if __name__ == '__main__':
    main()
