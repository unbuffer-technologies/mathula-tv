#!/bin/bash
# Installs the weekly crawl as a systemd USER timer (the server has no crontab): Sundays 03:00 UTC, catching up after downtime, log in logs/weekly_crawl.log.
# enable-linger lets the timer run when nobody is logged in.  Undo: systemctl --user disable --now mathula-weekly-crawl.timer
set -e
mkdir -p ~/.config/systemd/user ~/mathula-tv/logs
cat > ~/.config/systemd/user/mathula-weekly-crawl.service <<'EOF'
[Unit]
Description=Mathula TV weekly Isolezwe crawl and web index rebuild

[Service]
Type=oneshot
ExecStart=/bin/bash %h/mathula-tv/scripts/weekly_crawl.sh
StandardOutput=append:%h/mathula-tv/logs/weekly_crawl.log
StandardError=append:%h/mathula-tv/logs/weekly_crawl.log
EOF
cat > ~/.config/systemd/user/mathula-weekly-crawl.timer <<'EOF'
[Unit]
Description=Weekly Isolezwe crawl (Sunday 03:00 UTC)

[Timer]
OnCalendar=Sun *-*-* 03:00:00
Persistent=true
RandomizedDelaySec=600

[Install]
WantedBy=timers.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now mathula-weekly-crawl.timer
loginctl enable-linger "$USER"
systemctl --user list-timers mathula-weekly-crawl.timer --no-pager
