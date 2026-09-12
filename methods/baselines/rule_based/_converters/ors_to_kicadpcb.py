#!/usr/bin/env python3
"""ORS (orthoroute_gpu) -> kicad_pcb converter (text-based, no pcbnew).

Reads an .ORS file (gzip-compressed JSON) and the original unrouted .kicad_pcb,
and emits a routed .kicad_pcb by injecting (segment ...) / (via ...) tokens
just before the closing ')' of the kicad_pcb root S-expression.

KiCad 9 PCB files are S-expression text. This is a lossless append: every
existing token in the source PCB is preserved verbatim.
"""
import argparse
import gzip
import json
import re
import sys
import uuid
from pathlib import Path


def load_ors(path: Path):
    with open(path, 'rb') as f:
        head = f.read(2)
    f = gzip.open(path, 'rt', encoding='utf-8') if head == b'\x1f\x8b' else open(path, 'r', encoding='utf-8')
    with f:
        return json.load(f)


def parse_net_map(pcb_text: str):
    """Parse top-level (net N "NAME") declarations -> {name: id}."""
    nets = {}
    for m in re.finditer(r'\(net\s+(\d+)\s+"([^"]*)"\)', pcb_text):
        nid = int(m.group(1))
        name = m.group(2)
        if name and name not in nets:
            nets[name] = nid
    return nets


def fmt_seg(x1, y1, x2, y2, width, layer, net_id):
    return (
        '  (segment\n'
        f'    (start {x1} {y1})\n'
        f'    (end {x2} {y2})\n'
        f'    (width {width})\n'
        f'    (layer "{layer}")\n'
        f'    (net {net_id})\n'
        f'    (uuid "{uuid.uuid4()}")\n'
        '  )'
    )


def fmt_via(x, y, size, drill, layer_a, layer_b, net_id):
    return (
        '  (via\n'
        f'    (at {x} {y})\n'
        f'    (size {size})\n'
        f'    (drill {drill})\n'
        f'    (layers "{layer_a}" "{layer_b}")\n'
        f'    (net {net_id})\n'
        f'    (uuid "{uuid.uuid4()}")\n'
        '  )'
    )


def fmt_num(v):
    """Format mm -> string with reasonable precision; trims pointless zeros."""
    s = f'{float(v):.6f}'
    s = s.rstrip('0').rstrip('.')
    return s if s else '0'


def convert(ors_path: Path, base_pcb: Path, out_pcb: Path):
    ors = load_ors(ors_path)
    base_text = base_pcb.read_text(encoding='utf-8')
    net_map = parse_net_map(base_text)

    geom = ors.get('geometry', {})
    by_net = geom.get('by_net', {})
    all_tracks = geom.get('all_tracks') or []
    all_vias = geom.get('all_vias') or []

    # The orthoroute schema we have stores tracks/vias under by_net only.
    # If all_tracks is empty, flatten by_net.
    if not all_tracks and by_net:
        for net_id, ng in by_net.items():
            for t in ng.get('tracks', []) or []:
                tt = dict(t)
                tt['net'] = net_id
                all_tracks.append(tt)
            for v in ng.get('vias', []) or []:
                vv = dict(v)
                vv['net'] = net_id
                all_vias.append(vv)

    out_blocks = []
    skipped_tracks = 0
    skipped_vias = 0

    for t in all_tracks:
        layer = t.get('layer', 'F.Cu')
        s = t.get('start') or {}
        e = t.get('end') or {}
        x1 = float(s.get('x', 0.0)); y1 = float(s.get('y', 0.0))
        x2 = float(e.get('x', 0.0)); y2 = float(e.get('y', 0.0))
        width = float(t.get('width', 0.25))
        net_name = t.get('net') or t.get('net_id') or ''
        net_id = net_map.get(net_name, 0)
        if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
            skipped_tracks += 1
            continue
        out_blocks.append(fmt_seg(fmt_num(x1), fmt_num(y1), fmt_num(x2), fmt_num(y2),
                                  fmt_num(width), layer, net_id))

    for v in all_vias:
        pos = v.get('position') or {'x': v.get('x', 0.0), 'y': v.get('y', 0.0)}
        x = float(pos.get('x', 0.0)); y = float(pos.get('y', 0.0))
        diameter = float(v.get('diameter', 0.6))
        drill = float(v.get('drill', 0.3))
        layer_a = v.get('from_layer', 'F.Cu')
        layer_b = v.get('to_layer', 'B.Cu')
        net_name = v.get('net') or v.get('net_id') or ''
        net_id = net_map.get(net_name, 0)
        out_blocks.append(fmt_via(fmt_num(x), fmt_num(y), fmt_num(diameter),
                                  fmt_num(drill), layer_a, layer_b, net_id))

    # Inject before the final closing ')' of the kicad_pcb root.
    rstripped = base_text.rstrip()
    if not rstripped.endswith(')'):
        raise RuntimeError(f'Unexpected PCB tail in {base_pcb}: does not end with ")"')
    body = rstripped[:-1].rstrip()
    injected = body + '\n\n' + '\n'.join(out_blocks) + '\n)\n'

    out_pcb.parent.mkdir(parents=True, exist_ok=True)
    out_pcb.write_text(injected, encoding='utf-8')

    return {
        'tracks_in': len(all_tracks),
        'tracks_written': len(all_tracks) - skipped_tracks,
        'tracks_skipped': skipped_tracks,
        'vias_in': len(all_vias),
        'vias_written': len(all_vias) - skipped_vias,
        'nets_mapped': len(net_map),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ors')
    ap.add_argument('base_pcb')
    ap.add_argument('output')
    args = ap.parse_args()
    stats = convert(Path(args.ors), Path(args.base_pcb), Path(args.output))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
