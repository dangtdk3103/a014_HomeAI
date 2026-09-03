#!/usr/bin/env bash
set -u

bind_address="${DJANGO_BIND:-0.0.0.0:8000}"

while true; do
  echo "Starting Django at $(date) on ${bind_address}"
  python manage.py runserver "${bind_address}" --noreload
  exit_code=$?
  echo "Django exited with code ${exit_code}. Restarting in 2s."
  sleep 2
done
