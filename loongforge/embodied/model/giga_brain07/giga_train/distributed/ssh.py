import os

import paramiko


def ssh_copy_id(ips: list[str], username: str, password: str, port: int) -> None:
    """Copy local public key to multiple remote hosts via Paramiko.

    This generates a temporary script to invoke `ssh-copy-id` on each host.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    run_path = os.path.join(cur_dir, 'run.sh')
    authorized_keys = os.path.join(cur_dir, 'authorized_keys')
    clients = []
    for ip in ips:
        transport = paramiko.Transport(sock=(ip, port))
        transport.connect(username=username, password=password)
        client = paramiko.SSHClient()
        client._transport = transport
        clients.append(client)
    for i, client in enumerate(clients):
        cmd = f'cat /root/.ssh/id_rsa.pub >> {authorized_keys}'
        stdin, stdout, stderr = client.exec_command(cmd)
        print(ips[i], stdout.readlines())
    with open(run_path, 'w') as fn:
        fn.write('#!/bin/bash\n\n')
        fn.write(f'cp {authorized_keys} /root/.ssh/\n')
        for ip in ips:
            cmd = f'ssh-copy-id -i /root/.ssh/id_rsa.pub {username}@{ip} -p {port} -o "StrictHostKeyChecking no"'
            fn.write(f'{cmd}\n')
    for i, client in enumerate(clients):
        cmd = f'sh {run_path}'
        stdin, stdout, stderr = client.exec_command(cmd)
        print(ips[i], stdout.readlines())
    os.remove(authorized_keys)
    os.remove(run_path)
