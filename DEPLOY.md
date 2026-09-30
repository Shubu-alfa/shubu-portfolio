# Deploy shubzfolio.org (VPS + Neon + Cloudflare)

This guide matches the locked stack: small Ubuntu VPS, Neon PostgreSQL, nginx + gunicorn, Cloudflare DNS for `shubzfolio.org`.

## 1. Neon PostgreSQL

1. Create a free project at [https://neon.tech](https://neon.tech).
2. Copy the connection string (`DATABASE_URL`). It should look like:
   `postgresql://USER:PASSWORD@HOST/DB?sslmode=require`
3. Tables `chat_sessions` and `chat_messages` are created automatically on first app start via SQLAlchemy.

## 2. Generate secrets (on your laptop)

```bash
python -c "import secrets; print(secrets.token_hex(32))"
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('YOUR_ADMIN_PASSWORD'))"
```

Put them in the server `.env`:

```env
SECRET_KEY=...
ADMIN_PASSWORD_HASH=pbkdf2:sha256:...
DATABASE_URL=postgresql://...
```

## 3. Create the VPS

1. Create an Ubuntu 22.04/24.04 VPS (~$4–6/mo: Hetzner, DigitalOcean, etc.).
2. Note the public IPv4 address.
3. Firewall: allow SSH (22), HTTP (80), HTTPS (443) only.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip nginx git
```

## 4. App on the VPS

```bash
sudo mkdir -p /var/www/shubu-portfolio
sudo chown $USER:$USER /var/www/shubu-portfolio
cd /var/www/shubu-portfolio
git clone https://github.com/Shubu-alfa/shubu-portfolio.git .
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ensure `flan-alpaca-base/` (model weights) is present on the server (git LFS, scp, or rsync). Create `/var/www/shubu-portfolio/.env` with the secrets from step 2.

Quick check:

```bash
source venv/bin/activate
python -c "from db import init_db; init_db(); print('db ok')"
```

## 5. systemd service

Create `/etc/systemd/system/shubzfolio.service`:

```ini
[Unit]
Description=ShubzFolio Flask app
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/shubu-portfolio
EnvironmentFile=/var/www/shubu-portfolio/.env
ExecStart=/var/www/shubu-portfolio/venv/bin/gunicorn -b 127.0.0.1:8000 -w 1 --timeout 120 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo chown -R www-data:www-data /var/www/shubu-portfolio
sudo systemctl daemon-reload
sudo systemctl enable --now shubzfolio
sudo systemctl status shubzfolio
```

Use `-w 1` while the chatbot loads a local model in-process (avoids loading the model multiple times).

## 6. nginx reverse proxy

Create `/etc/nginx/sites-available/shubzfolio`:

```nginx
server {
    listen 80;
    server_name shubzfolio.org www.shubzfolio.org;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/shubzfolio /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

TLS options:

- **Cloudflare proxy (orange cloud):** set SSL/TLS mode to Full (Strict) and use an origin certificate, or
- **Direct TLS:** `sudo apt install certbot python3-certbot-nginx && sudo certbot --nginx -d shubzfolio.org -d www.shubzfolio.org`

## 7. Cloudflare DNS

In Cloudflare for `shubzfolio.org`:

| Type | Name | Content |
|------|------|---------|
| A | `@` | your VPS IPv4 |
| A or CNAME | `www` | VPS IP or `@` |

Wait for DNS to propagate, then open `https://shubzfolio.org`.

## 8. Verify

1. Open the site, send a chatbot message.
2. In Neon SQL editor:

```sql
SELECT * FROM chat_sessions ORDER BY created_at DESC LIMIT 5;
SELECT * FROM chat_messages ORDER BY created_at DESC LIMIT 20;
```

3. Visit `/trading` → login with your admin password → placeholder page loads.
4. Public pages (`/`, `/about`, chatbot) stay open without login.

## Notes

- Never commit `.env`.
- Locally, if `DATABASE_URL` is unset, the app uses `sqlite:///local_chat.db` so development still works.
- Rotate any tokens that were previously committed in git remotes or source files.
