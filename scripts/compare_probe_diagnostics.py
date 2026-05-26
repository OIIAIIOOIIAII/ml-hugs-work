# Compare multiple anchor probe/contact diagnostic JSON files.

import argparse
import csv
import json
from pathlib import Path


def flatten(prefix, data, out):
    for k, v in data.items():
        key = f'{prefix}{k}' if prefix else k
        if isinstance(v, dict):
            flatten(key + '.', v, out)
        elif isinstance(v, (int, float, str)) or v is None:
            out[key] = v


def main():
    parser = argparse.ArgumentParser(description='Compare probe/contact summaries across initialization variants.')
    parser.add_argument('--summary', action='append', required=True, help='Format name:path/to/summary.json')
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()
    rows = []
    keys = set(['name'])
    for item in args.summary:
        if ':' not in item:
            raise ValueError('--summary must be name:path')
        name, path = item.split(':', 1)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        row = {'name': name, 'path': path}
        flatten('', data, row)
        rows.append(row)
        keys.update(row.keys())
    ordered = ['name', 'path'] + sorted(k for k in keys if k not in ['name', 'path'])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    print(f'COMPARE_CSV={out}')


if __name__ == '__main__':
    main()
