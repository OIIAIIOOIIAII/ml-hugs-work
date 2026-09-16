#!/usr/bin/env bash
# Download RICH assets from an authorized URL list with resumable parallel wget.
#
# Usage:
#   bash download_rich_parallel.sh --urls rich_test_urls.txt --output-dir /path/to/RICH/test --jobs 3
#
# Each non-empty, non-comment line in --urls is one URL copied from the RICH
# download portal or its official authorized download script.  Authentication
# is deliberately delegated to the user's local wget configuration / netrc;
# never put credentials in this script or the URL list.

set -euo pipefail

url_file=""
output_dir=""
parallel_jobs=3

usage() {
    sed -n '2,11p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --urls)
            url_file=${2:?missing value for --urls}
            shift 2
            ;;
        --output-dir)
            output_dir=${2:?missing value for --output-dir}
            shift 2
            ;;
        --jobs)
            parallel_jobs=${2:?missing value for --jobs}
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$url_file" && -f "$url_file" ]] || { printf '%s\n' 'Provide an existing --urls file.' >&2; exit 2; }
[[ -n "$output_dir" ]] || { printf '%s\n' 'Provide --output-dir.' >&2; exit 2; }
[[ "$parallel_jobs" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--jobs must be a positive integer.' >&2; exit 2; }
command -v wget >/dev/null || { printf '%s\n' 'wget is required.' >&2; exit 127; }

mkdir -p "$output_dir"
mapfile -t urls < <(sed -E '/^[[:space:]]*(#|$)/d; s/^[[:space:]]+//; s/[[:space:]]+$//' "$url_file")
(( ${#urls[@]} > 0 )) || { printf '%s\n' 'The URL list has no downloadable entries.' >&2; exit 2; }

printf 'Downloading %d authorized RICH assets into %s with %d concurrent jobs.\n' "${#urls[@]}" "$output_dir" "$parallel_jobs"
printf '%s\n' 'wget --continue resumes partial files. A small HTML result usually means the portal authorization was not accepted.'

download_one() {
    local download_url=$1
    (
        cd "$output_dir"
        wget --continue --content-disposition --trust-server-names --netrc \
            --timeout=45 --tries=0 --waitretry=10 --retry-connrefused \
            "$download_url"
    )
}

for download_url in "${urls[@]}"; do
    download_one "$download_url" &
    while (( $(jobs -rp | wc -l) >= parallel_jobs )); do
        wait -n
    done
done
wait

printf '%s\n' 'Download processes completed. Verify archive checksums from the RICH portal before extraction.'
