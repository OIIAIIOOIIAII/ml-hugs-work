#!/usr/bin/env python3
"""Generate all RICH sole labels and raw/cropped frustum masks; resume by hashes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_streaming.rich_supervision import prepare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rich-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--size',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(prepare(a.rich_root,a.output,a.template,a.workers,a.size),indent=2))
if __name__=='__main__': main()
