
def require_copy(src, dst, follow_symlinks=True):
    """
    Request a copy using subprocess executing the 'cp' command.
    If dst does not exist, it will be created.

    Returns:
        Path to directory.
    """
    from os.path import exists
    import subprocess
    from subprocess import CalledProcessError

    if not exists(dst):
        parameter = '-r'
        if follow_symlinks:
            parameter += 'L'

        try:
            subprocess.run(['cp', parameter, src, dst], check=True)
        except CalledProcessError:
            if not exists(dst):
                raise

    return dst
