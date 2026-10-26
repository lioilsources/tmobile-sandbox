# TMOBILE_SANDBOX_PROVISION_PLAN.md (v2 — ruční VM z ISO)

Provisioning sandboxu T‑Mobile Business Cloud GPUaaS — 1× NVIDIA L40S (48 GB), VM Ubuntu 24.04, 1 TB superfast disk.
Cíl: do 2 dnů mít vLLM + ComfyUI + Whisper + monitoring, stažené modely a smoke testy pro tři dema
(call centrum, AI API, dokumenty). Vše idempotentní (skripty v repu `tmobile-sandbox/`), aby šel stroj kdykoli znovu postavit.

**Změna oproti v1:** T‑Mobile šablony leží na standardním disk tieru, účet má superfast → šablony nejdou použít.
OS instaluji sám z vlastního ISO. Důsledky: driver není předinstalovaný, disk layout určuji sám, přibývá QEMU guest agent,
a musím ověřit, jak se GPU k ručně vytvořené VM připojuje (passthrough vs vGPU).

---

## −1. Otázky na devopsáka T‑Mobile (před vytvořením VM)

1. **Jak se L40S přiřadí k ručně vytvořené VM?** Je ve wizardu volba GPU, nebo ji přidáte vy?
2. ~~PCI passthrough, nebo vGPU?~~ **Potvrzeno 24. 10.: PCI passthrough.** Stačí `nvidia-driver-570-server` z Ubuntu repa, žádný GRID/licenční server.
3. **Kvóta účtu** – ve wizardu Medium‑2 a výš svítí varování; povolí kvóta 1 TB superfast + ≥ 46 GB RAM + ≥ 8 vCPU?
4. **Egress** – jsou HF / GitHub / PyPI / Docker Hub / NGC povolené ven z VM?
5. **Veřejná IP vs NAT** – dostane VM veřejnou IP, nebo jde port 22 přes Virtual Router?

---

## 0. Vytvoření VM z ISO (portál)

1. Stáhnout `ubuntu-24.04.x-live-server-amd64.iso` (ověřit SHA256).
2. Portál → **Library** → upload ISO.
3. **Compute → Create → Custom** (ne Simple Server):
   - vCPU: max, co kvóta dovolí (ideálně 8–16)
   - RAM: ≥ 46 GB (vLLM Qwen3‑32B FP8 + ComfyUI potřebují RAM na načítání)
   - Disk: **1 TB, tier superfast**
   - Boot/CD‑ROM: nahrané ISO
   - GPU: podle odpovědi na otázku 1
4. Spustit VM, otevřít **konzoli v portálu**, projít installer:
   - Minimal install (bez snapů navíc)
   - **Storage: vypnout „Set up this disk as an LVM group"**, nebo pak rozšířit (viz 1a). Výchozí LVM dá rootu jen ~100 GB!
   - Swap: 8 GB stačí (modely se mapují z disku, ne ze swapu)
   - **SSH: Install OpenSSH server**, import klíče z GitHubu (`lioilsources`) nebo vložit `id_ed25519_tcloud.pub`
   - Uživatel: `<user>`
5. Po instalaci **odpojit ISO** v portálu, reboot.
6. **Networking** – veřejná IP nebo NAT 22 → VM; firewall: TCP 22 jen z mé IP; vše ostatní zavřené (Grafana/Open WebUI půjde přes SSH tunel).
7. Lokálně `~/.ssh/config`:
   ```
   Host tcloud
       HostName <veřejná-IP>
       User <user>
       IdentityFile ~/.ssh/id_ed25519_tcloud
   ```

---

## 1. První boot: inventura + hypervizor

```bash
cat /etc/os-release
lspci | grep -i nvidia                   # !!! bez výstupu = GPU není přiřazena → devopsák
nproc; free -h; lsblk; df -h /
ip a; cat /etc/resolv.conf
curl -sI https://huggingface.co | head -1
curl -sI https://github.com | head -1
curl -sI https://download.docker.com | head -1
curl -sI https://nvcr.io | head -1
```

### 1a. Disk na celý 1 TB (pokud installer nechal LVM)
```bash
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
df -h /                                   # očekávám ~980 GB
```

### 1b. QEMU guest agent (nové)
```bash
sudo apt update && sudo apt install -y qemu-guest-agent
sudo systemctl enable --now qemu-guest-agent
ls -l /dev/virtio-ports/org.qemu.guest_agent.0 && systemctl is-active qemu-guest-agent
```
Bez agenta portál nevidí IP VM a shutdown z UI = hard reset.

### 1c. Základ
```bash
sudo apt install -y build-essential git tmux htop nvtop curl jq rclone unzip python3-venv python3-pip
sudo apt install -y linux-headers-$(uname -r)
sudo timedatectl set-timezone Europe/Prague
```

### 1d. NVIDIA driver (povinné, není předinstalovaný; PCI passthrough potvrzen)
```bash
lspci -nnk | grep -A3 -i nvidia                 # musí ukázat L40S [10de:26b9] a "Kernel driver in use: nouveau" nebo nic
sudo apt install -y nvidia-driver-570-server   # nebo novější server řada
sudo reboot
nvidia-smi                                      # L40S, 48 GB, driver ≥ 570
```
Pokud po rebootu `nvidia-smi` nic: `dmesg | grep -i nvidia` — u passthrough bývá problém jen Secure Boot (nepodepsaný modul) → v installeru/VM Secure Boot vypnout, nebo `mokutil`.

### 1e. Docker + NVIDIA Container Toolkit
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```
Docker data‑root přesunout na `/srv/docker` (viz 2).

### 1f. **SNAPSHOT „base"** v portálu (nové)
OS + agent + driver + Docker = stav, ke kterému se vracím při jakékoli katastrofě. U VM zadarmo.

---

## 2. Rozvržení disku (1 TB)

```
/srv/hf          ~330 GB   HF cache (HF_HOME) — modely pro vLLM, Whisper
/srv/comfy       ~150 GB   ComfyUI + models (checkpoints, VAE, LoRA, Wan, ACE‑Step)
/srv/docker      ~100 GB   docker data‑root (images: vllm, comfy, grafana…)
/srv/work        ~100 GB   repa, data pro dema, výstupy, benchmarky
/srv/backup       ~50 GB   staging pro rclone na domácí NAS
rezerva          ~250 GB
```
```bash
sudo mkdir -p /srv/{hf,comfy,docker,work,backup} && sudo chown -R $USER /srv
echo 'export HF_HOME=/srv/hf' >> ~/.bashrc
# /etc/docker/daemon.json: {"data-root": "/srv/docker", "runtimes": {...}}
```

---

## 3. Modely z Hugging Face (~310 GB, stahovat v tmux)

```bash
pip install -U "huggingface_hub[cli]" && huggingface-cli login
```
| Model | Účel | ~GB |
|---|---|---|
| `Qwen/Qwen3-32B-FP8` | hlavní LLM (dema 1, 2, 3) | 35 |
| `Qwen/Qwen3-30B-A3B` | rychlý MoE pro souběh / API tier | 60 |
| `Qwen/Qwen3-VL-32B-Instruct` (ověřit aktuální název) | dokumenty / OCR‑VL | 65 |
| `Qwen/Qwen3-Embedding-0.6B` | embeddings pro API | 2 |
| `Systran/faster-whisper-large-v3-turbo` | čeština ASR | 2 |
| `black-forest-labs/FLUX.1-schnell` | obrázky (Apache 2.0, **ne dev**) | 24 |
| `Qwen/Qwen-Image` | obrázky, alternativa | 40 |
| `Wan-AI/Wan2.2-T2V-A14B` + `Wan2.2-I2V-A14B` | video | 60 |
| `stabilityai/stable-diffusion-xl-base-1.0` + VAE | SDXL (realeyes LoRA) | 7 |
| `ACE-Step/ACE-Step-v1-3.5B` | hudba/SFX | 8 |

Před stažením: `huggingface-cli repo info <id>` — Qwen přejmenovává. Licence: FLUX.1‑dev je non‑commercial → pro T‑Mobile schnell / Qwen‑Image / Wan (Apache 2.0).

---

## 4. vLLM — VRAM profily (48 GB neudrží vše naráz)

| Profil | Co běží | VRAM |
|---|---|---|
| A „LLM“ | Qwen3‑32B‑FP8 (ctx 32k) | ~44 |
| B „API mix“ | Qwen3‑30B‑A3B + embeddings + Whisper | ~40 |
| C „Dokumenty“ | Qwen3‑VL‑32B (FP8) | ~44 |
| D „Media“ | ComfyUI (FLUX‑schnell / Wan) + Whisper | ~36 |

Přepínání přes controller‑manager z [[aistack]] (docker.sock). **AiStack je buildnutý na ARM (Spark) → ověřit x86_64 Dockerfiles jako první úkol dne 2.**
```bash
docker run -d --gpus all --name vllm -p 8000:8000 -v /srv/hf:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-32B-FP8 --max-model-len 32768 --gpu-memory-utilization 0.92
vllm bench serve --model Qwen/Qwen3-32B-FP8 --num-prompts 200 --request-rate 8
```

---

## 5. ComfyUI + custom nodes

```bash
cd /srv/comfy && git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager
```
Nodes: ComfyUI‑Manager, KJNodes, WanVideoWrapper, VideoHelperSuite, rgthree, Impact‑Pack, ACE‑Step nodes, GGUF loader, ControlNet aux, IPAdapter plus. Workflow JSONy přenést ze Sparku (realeyes, Tsumiki karta, anime příběhy).

---

## 6. Whisper + Open WebUI + monitoring

- `faster-whisper-server` (CTranslate2) na :8001, čeština, `large-v3-turbo`
- Open WebUI na :3000 proti vLLM (pro demo UI)
- Monitoring **od prvního dne**: `nvidia/dcgm-exporter` + Prometheus + Grafana (dashboard 12239) + vLLM `/metrics`. Utilizace GPU přes den je hlavní číslo do reportu.
- Vše přes SSH tunel (`ssh -L 3000:localhost:3000 -L 3001:localhost:3001 tcloud`), žádné otevřené porty ven.

---

## 7. Zálohy + repo

- Repo `tmobile-sandbox/`: `00_inventory.sh`, `10_base.sh`, `20_driver.sh`, `30_docker.sh`, `40_models.sh`, `50_vllm/`, `60_comfy/`, `70_monitoring/`
- `rclone sync /srv/work nas:tmobile/work` denně (cron); workflow JSONy, benchmarky, Grafana dashboardy, syntetická data
- Modely nezálohovat (jdou znovu stáhnout), ale uložit **seznam + hashe**
- Snapshot „base“ (1f) + snapshot „full“ po dni 3

---

## 8. Dema (data)

1. **Call centrum** – nemám české nahrávky → syntetika: LLM napíše scénáře hovorů, TTS (dvě hlasy) namluví, Whisper přepíše, Qwen3 klasifikuje; metriky: h audia / h GPU, Kč/1000 hovorů
2. **AI API** – AiStack gateway, 3 API klíče (tenant A/B/C), per‑token účtování, Grafana per tenant; load test souběhu
3. **Dokumenty** – 50 faktur/smluv (vlastní + veřejné vzory) → Qwen3‑VL → JSON; metriky: stran/h, přesnost polí

---

## Harmonogram

| Den | Co |
|---|---|
| 0 (so/ne) | Otázky devopsákovi (−1), upload ISO, VM z ISO, SSH, síť |
| 1 dopo | Inventura, `lspci`, disk na 1 TB, guest agent, driver, Docker, **snapshot base**; spustit stahování (tmux) |
| 1 odpo | vLLM smoke test na prvním modelu, ComfyUI + nodes |
| 2 | Monitoring, Whisper, AiStack x86 build + nasazení, profily A–D ověřené v nvidia‑smi |
| 3 | Workflow JSONy, Open WebUI, první benchmark (`vllm bench serve`, obrázků/h), **snapshot full** |
| 4+ | Dema 1–3 podle `TMOBILE_PITCH` — data, UI, čísla do reportu |

## Rizika
- GPU není ve wizardu (jak se passthrough k VM přiřadí – support) → blokuje den 1, proto otázky −1 hned
- Secure Boot ve VM → nvidia modul se nenahraje; vypnout před instalací driveru
- Kvóta nedovolí 1 TB nebo RAM → devopsák
- Egress filtrovaný → whitelist HF/GitHub/PyPI/Docker Hub/NGC
- LVM 100 GB past → 1a
- AiStack ARM → x86 build
- Stroj za měsíc zmizí → repo + rclone + seznam modelů
