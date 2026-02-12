
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.utils
~~~~~~~~~~~~~

Some utility functions for globato. Taken from cudem.utils

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import shutil
import subprocess

from tqdm import tqdm

# System Command Functions
cmd_exists = lambda x: any(os.access(os.path.join(path, x), os.X_OK) 
                           for path in os.environ['PATH'].split(os.pathsep))


def run_cmd(cmd, data_fun=None, verbose=False, cwd='.'):
    """Run a system command while optionally passing data.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """
    
    out = None
    cols, _ = shutil.get_terminal_size()
    width = cols - 55
    
    with tqdm(desc=f'`{cmd.rstrip()[:width]}...`', leave=verbose) as pbar:
        pipe_stdin = subprocess.PIPE if data_fun is not None else None

        p = subprocess.Popen(
            cmd, shell=True, stdin=pipe_stdin, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, close_fds=True, cwd=cwd
        )

        if data_fun is not None:
            if verbose:
                echo_msg('Piping data to cmd subprocess...')
            data_fun(p.stdin)
            p.stdin.close()

        io_reader = io.TextIOWrapper(p.stderr, encoding='utf-8')
        while p.poll() is None:
            err_line = io_reader.readline()
            if verbose and err_line:
                pbar.write(err_line.rstrip())
                sys.stderr.flush()
            pbar.update()

        out = p.stdout.read()
        p.stderr.close()
        p.stdout.close()
        
        if verbose:
            echo_msg(f'Ran cmd {cmd.rstrip()} and returned {p.returncode}')
        
    return out, p.returncode


def yield_cmd(cmd, data_fun=None, verbose=False, cwd='.'):
    """Yield output from a system command.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """
    
    if verbose: echo_msg(f'Running cmd {cmd.rstrip()}...')
    
    pipe_stdin = subprocess.PIPE if data_fun is not None else None
    
    p = subprocess.Popen(
        cmd, shell=True, stdin=pipe_stdin, stdout=subprocess.PIPE,
        close_fds=True, cwd=cwd
    )
    
    if data_fun is not None:
        if verbose: echo_msg('Piping data to cmd subprocess...')
        data_fun(p.stdin)
        p.stdin.close()

    while p.poll() is None:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        yield line
        
    p.stdout.close()
    if verbose:
        echo_msg(f'Ran cmd {cmd.rstrip()}, returned {p.returncode}.')

        
def cmd_check(cmd_str, cmd_vers_str):
    """check system for availability of 'cmd_str'"""
    
    if cmd_exists(cmd_str): 
        cmd_vers, status = run_cmd(f'{cmd_vers_str}')
        return cmd_vers.rstrip()
    return b"0"
