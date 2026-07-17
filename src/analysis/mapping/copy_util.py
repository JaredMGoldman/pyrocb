import paramiko
import os

def upload_simplified(local_file, remote_file, hostname, username = os.environ['USER']):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Paramiko automatically searches ~/.ssh/ and queries your ssh-agent here:
        print(f"Connecting seamlessly to {hostname}...")
        ssh.connect(hostname=hostname, username=username,key_filename=os.path.join(os.environ['HOME'], '.ssh', 'id_rsa.pub')) 
        
        sftp = ssh.open_sftp()
        sftp.put(local_file, remote_file)
        sftp.close()
        print("[+] Upload successful.")
        
    except Exception as e:
        print(f"[-] Connection failed: {e}")
    finally:
        ssh.close()

import os
import paramiko

def create_remote_symlink(remote_dir, source_filename, symlink_name="latest.html", hostname="your.server.ip", username="ubuntu"):
    """
    Connects to a remote server using an SSH keypair and atomically creates/overwrites
    a symbolic link pointing to a specified target file.
    """
    
    # Establish absolute targeting paths for the remote Linux filesystem
    absolute_source_file = os.path.join(remote_dir, source_filename)
    absolute_symlink_target = os.path.join(remote_dir, symlink_name)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to remote host: {hostname}...")
        ssh.connect(hostname=hostname, username=username)
        
        # -s: creates a symbolic link instead of a hard link
        # -f: forces an atomic overwrite if 'latest.html' already exists (prevents FileExists errors)
        ln_command = f"ln -sf {absolute_source_file} {absolute_symlink_target}"
        print(f"Executing remote command: {ln_command}")
        
        # Execute the command on the remote shell channel
        stdin, stdout, stderr = ssh.exec_command(ln_command)
        
        # Read back error buffers to catch path formatting issues or permission blocks
        error_msg = stderr.read().decode()
        if error_msg:
            print(f"[-] Remote symlink failed: {error_msg.strip()}")
        else:
            print(f"[+] Successfully created remote symlink: {symlink_name} -> {source_filename}")
            
    except Exception as e:
        print(f"[-] Connection or execution error: {e}")
    finally:
        # Cleanly drop network socket attachments
        ssh.close()
