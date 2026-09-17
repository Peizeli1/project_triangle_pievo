#!/bin/bash
set -euo pipefail
echo "ResearchClawBench credential setup (stored in macOS Keychain; input is hidden)."
echo
printf "SERPER_KEY: "; IFS= read -r -s SERPER_KEY; echo
printf "JINA_KEY: "; IFS= read -r -s JINA_KEY; echo
printf "MINERU_TOKEN: "; IFS= read -r -s MINERU_TOKEN; echo
if [[ -z "" || -z "" || -z "" ]]; then echo "One or more values were empty; nothing was saved."; exit 2; fi
security add-generic-password -U -a "michael" -s "ai4ai.researchclaw.serper" -w "" >/dev/null
security add-generic-password -U -a "michael" -s "ai4ai.researchclaw.jina" -w "" >/dev/null
security add-generic-password -U -a "michael" -s "ai4ai.researchclaw.mineru" -w "" >/dev/null
unset SERPER_KEY JINA_KEY MINERU_TOKEN
echo
echo "Saved all three credentials to macOS Keychain. You can close this window."
read -r -p "Press Enter to close..." _
