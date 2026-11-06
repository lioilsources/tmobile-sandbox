# comfy_relay — přístup k ComfyUI z domácí LAN přes M2

Reverse proxy na M2: z LAN (`:18188`) na ComfyUI tunel `127.0.0.1:8188` (tcloud-tunnel, VPN). Každý požadavek musí mít `Authorization: Bearer <token>`. Propouští jen `/prompt`, `/history/*`, `/view`, `/upload/image` a `/system_stats`, tělo max. 32 MB. Hlavičku `Authorization` dál k ComfyUI neposílá.

Token je v `~/.config/comfy-relay/token` na M2 (práva 600, vytvoří se `openssl rand -hex 24`). Klient ho dostane do proměnné `COMFY_TOKEN`, PartyCockpit pak `COMFY_URL=http://<LAN IP M2>:18188`.

```
cd 92_comfy_relay && go build -o ~/bin/comfy_relay .
nohup ~/bin/comfy_relay -listen :18188 -token-file ~/.config/comfy-relay/token > ~/.config/comfy-relay/relay.log 2>&1 &
```

Po akci proxy zastavit (`pkill comfy_relay`).
