#!/usr/bin/env bash
# Download SMPL v1.1.0 (Python) from smpl.is.tue.mpg.de.
# Credentials are read interactively; nothing is persisted to history or env.

set -euo pipefail

DOMAIN="smpl"
HOST="https://smpl.is.tue.mpg.de"
DL_HOST="https://download.is.tue.mpg.de"
SFILE="SMPL_python_v.1.1.0.zip"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/data/smpl"
TMP_DIR="$(mktemp -d -t smpl-dl-XXXXXX)"
COOKIE_JAR="$TMP_DIR/cookies.txt"
ZIP_PATH="$TMP_DIR/$SFILE"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TARGET_DIR"

echo "==> SMPL v1.1.0 downloader"
echo "    target dir : $TARGET_DIR"
echo "    file       : $SFILE"
echo

# Credential sources, in priority order:
#   1. SMPL_CREDS_FILE env var pointing at a file with two lines: user, pass
#   2. SMPL_USER + SMPL_PASS env vars
#   3. interactive read from controlling TTY (only works with a real terminal)
SMPL_USER="${SMPL_USER:-}"
SMPL_PASS="${SMPL_PASS:-}"

if [[ -n "${SMPL_CREDS_FILE:-}" ]]; then
  if [[ ! -r "$SMPL_CREDS_FILE" ]]; then
    echo "ERROR: SMPL_CREDS_FILE=$SMPL_CREDS_FILE not readable" >&2
    exit 1
  fi
  SMPL_USER=$(sed -n '1p' "$SMPL_CREDS_FILE")
  SMPL_PASS=$(sed -n '2p' "$SMPL_CREDS_FILE")
fi

if [[ -z "$SMPL_USER" || -z "$SMPL_PASS" ]]; then
  if [[ -r /dev/tty ]]; then
    read -r -p "SMPL username (email): " SMPL_USER < /dev/tty
    read -r -s -p "SMPL password: " SMPL_PASS < /dev/tty
    echo
  fi
fi

if [[ -z "$SMPL_USER" || -z "$SMPL_PASS" ]]; then
  echo "ERROR: no credentials provided." >&2
  echo "       Set SMPL_CREDS_FILE=/path/to/file (line1=user, line2=pass)" >&2
  echo "       or run from an interactive terminal." >&2
  exit 1
fi

echo "==> Logging in ..."
HTTP_CODE=$(curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -o "$TMP_DIR/login.html" -w "%{http_code}" \
  --data-urlencode "username=$SMPL_USER" \
  --data-urlencode "password=$SMPL_PASS" \
  --data-urlencode "commit=Log in" \
  "$HOST/login.php")

unset SMPL_PASS

if ! grep -qiE 'logout|sign out' "$TMP_DIR/login.html"; then
  echo "ERROR: login does not appear successful (http $HTTP_CODE)." >&2
  echo "       Check username/password, or visit the site to clear captcha." >&2
  exit 2
fi
echo "    login OK"

echo "==> Downloading $SFILE ..."
curl -sS -L --fail -b "$COOKIE_JAR" -c "$COOKIE_JAR" \
  --referer "$HOST/download.php" \
  -o "$ZIP_PATH" \
  "$DL_HOST/download.php?domain=$DOMAIN&sfile=$SFILE&resume=1"

# Sanity check: must be a real zip, not an HTML redirect to login.
if ! file "$ZIP_PATH" | grep -qi 'zip archive'; then
  echo "ERROR: downloaded file is not a zip archive. Server returned:" >&2
  head -c 400 "$ZIP_PATH" >&2 || true
  echo >&2
  exit 3
fi
echo "    downloaded $(du -h "$ZIP_PATH" | cut -f1)"

echo "==> Extracting ..."
EXTRACT_DIR="$TMP_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"

# Locate the neutral pkl. The archive layout is typically:
#   SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
NEUTRAL_PKL=$(find "$EXTRACT_DIR" -type f -iname '*neutral*.pkl' | head -n1)
if [[ -z "$NEUTRAL_PKL" ]]; then
  echo "ERROR: could not find neutral .pkl inside archive. Contents:" >&2
  find "$EXTRACT_DIR" -maxdepth 4 -type f >&2
  exit 4
fi

cp -f "$NEUTRAL_PKL" "$TARGET_DIR/SMPL_NEUTRAL.pkl"
echo "    -> $TARGET_DIR/SMPL_NEUTRAL.pkl"

# Optional: also stage MALE/FEMALE if present (HUGS README shows them in the layout).
for sex in male female; do
  match=$(find "$EXTRACT_DIR" -type f -iname "*${sex}*.pkl" | head -n1 || true)
  if [[ -n "$match" ]]; then
    upper=$(printf '%s' "$sex" | tr '[:lower:]' '[:upper:]')
    cp -f "$match" "$TARGET_DIR/SMPL_${upper}.pkl"
    echo "    -> $TARGET_DIR/SMPL_${upper}.pkl"
  fi
done

echo
echo "==> Done. data/smpl now contains:"
ls -lh "$TARGET_DIR"
