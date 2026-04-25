import base64
import logging
import re
from urllib.parse import urljoin

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

logger = logging.getLogger(__name__)

GENAI_BASE_URL = "https://genai.shanghaitech.edu.cn"
IDS_BASE_URL = "https://ids.shanghaitech.edu.cn"


class LoginError(Exception):
    pass


def encrypt_password(password: str, salt: str) -> str:
    prefix = b"Nu1L" * 16
    combined = prefix + password.encode()
    iv = b"Nu1LNu1LNu1LNu1L"
    key = salt.encode()
    padded = pad(combined, AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode()


def _collect_data(html: str, name: str, end_tag: str = "/>") -> str:
    if name == "pwdEncryptSalt":
        start = html.find(f'id="{name}"')
    else:
        start = html.find(f'name="{name}"')
    if start == -1:
        return ""
    end = html.find(end_tag, start)
    raw = html[start:end]
    val_start = raw.find('value="') + 7
    val_end = raw.find('"', val_start)
    return raw[val_start:val_end]


def _get_service_url(html: str) -> str | None:
    m = re.search(r'var service = \["(.*?)"', html)
    return m.group(1) if m else None


def login_genai(student_id: str, password: str) -> str:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    })

    # Step 1: GET GenAI login entry
    resp = session.get(f"{GENAI_BASE_URL}/htk/user/login", allow_redirects=False)

    if 300 <= resp.status_code < 400:
        ids_login_url = urljoin(f"{GENAI_BASE_URL}/htk/user/login", resp.headers["Location"])
    else:
        service_url = _get_service_url(resp.text)
        if not service_url:
            raise LoginError("Failed to determine IDS service URL from GenAI login page")
        ids_login_url = f"{IDS_BASE_URL}/authserver/login?service={service_url}"

    # Step 2: GET IDS login page (follow redirects)
    ids_resp = session.get(ids_login_url)
    ids_html = ids_resp.text

    # Step 3: Parse hidden form fields
    lt = _collect_data(ids_html, "lt")
    execution = _collect_data(ids_html, "execution")
    salt = _collect_data(ids_html, "pwdEncryptSalt")

    if not salt:
        raise LoginError("Failed to get pwdEncryptSalt from IDS login page")
    if not execution:
        raise LoginError("Failed to get execution from IDS login page")

    # Step 4: Encrypt password & POST
    encrypted_pwd = encrypt_password(password, salt)

    form_data = {
        "username": student_id,
        "password": encrypted_pwd,
        "lt": lt,
        "dllt": "generalLogin",
        "execution": execution,
        "_eventId": "submit",
        "rmShown": "1",
    }

    logger.info("Posting login form to IDS at: %s", ids_resp.url)
    post_resp = session.post(
        ids_resp.url,
        data=form_data,
        allow_redirects=False,
    )

    # Follow redirects manually to catch ?token= in Location
    max_redirects = 10
    for _ in range(max_redirects):
        if not (300 <= post_resp.status_code < 400):
            break
        location = post_resp.headers.get("Location", "")
        resolved = urljoin(post_resp.url, location)

        if "?token=" in resolved or "&token=" in resolved:
            from urllib.parse import urlparse, parse_qs
            token = parse_qs(urlparse(resolved).query).get("token", [None])[0]
            if token:
                logger.info("Login successful, token obtained")
                return token

        post_resp = session.get(resolved, allow_redirects=False)

    # Check for login failure in response body
    body = post_resp.text if post_resp.status_code == 200 else ""
    if any(kw in body for kw in ("authError", "用户名或密码", "incorrectPassword")):
        raise LoginError("Username or password is incorrect")

    # Last chance: check final URL for token
    if "?token=" in post_resp.url or "&token=" in post_resp.url:
        from urllib.parse import urlparse, parse_qs
        token = parse_qs(urlparse(post_resp.url).query).get("token", [None])[0]
        if token:
            logger.info("Login successful, token obtained")
            return token

    raise LoginError("Login flow completed but failed to extract token")
