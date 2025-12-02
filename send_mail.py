import os
import sys
import socket
import smtplib
import ssl
import logging
import traceback
from types import SimpleNamespace
from email.message import EmailMessage
from typing import List, Optional

def _stream_writer(logger: logging.Logger, level: int) -> SimpleNamespace:
    def write(msg: str) -> None:
        msg = msg.rstrip('\n')
        if msg:
            logger.log(level, msg)
    def flush() -> None:
        pass
    return SimpleNamespace(write=write, flush=flush)

logger = logging.getLogger('send_mail')
logger.setLevel(logging.DEBUG)

run_handler = logging.FileHandler('run.log', encoding='utf-8')
run_handler.setLevel(logging.DEBUG)

email_handler = logging.FileHandler('email.log', encoding='utf-8')
email_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
run_handler.setFormatter(formatter)
email_handler.setFormatter(formatter)

logger.addHandler(run_handler)
logger.addHandler(email_handler)

console = logging.StreamHandler(sys.__stdout__)
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

sys.stdout = _stream_writer(logger, logging.INFO)
sys.stderr = _stream_writer(logger, logging.ERROR)

def _ipv4_targets(host: str, port: int):
    try:
        infos = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in infos:
            yield info[4][0]
    except Exception:
        logger.debug("IPv4 resolution failed for %s:%s", host, port, exc_info=True)
    yield host

def _try_smtp(hosts, port, timeout, use_tls, smtp_server, context, username, password, msg):
    last_error = None
    for target in hosts:
        try:
            if use_tls:
                with smtplib.SMTP(target, port, timeout=timeout) as smtp:
                    smtp.set_debuglevel(1)
                    smtp.ehlo()
                    smtp.starttls(context=context, server_hostname=smtp_server)
                    smtp.ehlo()
                    if username and password:
                        smtp.login(username, password)
                    smtp.send_message(msg)
                    return
            else:
                with smtplib.SMTP_SSL(target, port, context=context, timeout=timeout) as smtp:
                    smtp.set_debuglevel(1)
                    if username and password:
                        smtp.login(username, password)
                    smtp.send_message(msg)
                    return
        except Exception as exc:
            last_error = exc
            logger.warning("Connect attempt to %s:%s failed: %s", target, port, exc)
    raise TimeoutError(f"All SMTP targets failed for {smtp_server}:{port}") from last_error

def send_email(
    subject: str,
    body: str,
    to_addrs: List[str],
    from_addr: str,
    smtp_server: str = 'smtp.gmail.com',
    port: int = 587,
    username: Optional[str] = None,
    password: Optional[str] = None,
    use_tls: bool = True,
    attachments: Optional[List[str]] = None,
    timeout: int = 15
) -> bool:
    username = username or from_addr
    password = password or os.environ.get('SMTP_PASSWORD')

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = ', '.join(to_addrs)
    msg.set_content(body)

    if attachments:
        for path in attachments:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                filename = os.path.basename(path)
                msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=filename)
            except Exception:
                logger.exception("Failed to attach file: %s", path)

    try:
        hosts = list(_ipv4_targets(smtp_server, port))
        context = ssl.create_default_context()
        _try_smtp(
            hosts=hosts,
            port=port,
            timeout=timeout,
            use_tls=use_tls,
            smtp_server=smtp_server,
            context=context,
            username=username,
            password=password,
            msg=msg,
        )
        logger.info("Email sent to %s subject=%s", to_addrs, subject)
        return True
    except Exception:
        logger.error("Failed to send email: %s", traceback.format_exc())
        return False

if __name__ == '__main__':
    success = send_email(
        subject='Test message',
        body='Hello from Python.',
        to_addrs=['ramtahor@walla.com'],
        from_addr='ramtahor69@gmail.com',
        smtp_server='smtp.gmail.com',
        port=587,
        username='ramtahor',
        password='ramta1986',
        use_tls=True,
        attachments=None,
        timeout=10,
    )
    if success:
        print('Sent')
        sys.exit(0)
    else:
        print('Failed')
        logger.error("Exiting with failure")
        sys.exit(1)