import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_status_email(status, details, recipient_email):
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # 16-character Google App Password
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"Script Execution Report: {status}"

    body = f"""
    The Python script has finished execution.
    
    Status: {status}
    Details: {details}
    
    This is an automated message.
    """
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls() # Secure the connection
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")