# systemd 单元模板（可选）

将以下示例保存为 `/etc/systemd/system/boyan-gate.service` / `boyan-brain.service`，并按需调整路径：

```ini
[Unit]
Description=Boyan v2 Gateway Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/Boyan
EnvironmentFile=/root/Boyan/.env
ExecStart=/root/Boyan/scripts/run_gate.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now boyan-gate.service
```
