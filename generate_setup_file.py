import subprocess

def conda_to_install_requires(env_name=None):
    cmd = ["conda", "list", "--export"]
    if env_name:
        cmd += ["-n", env_name]

    output = subprocess.check_output(cmd, text=True).splitlines()
    requirements = []

    for line in output:
        if line.startswith("#") or "conda" in line:
            continue
        parts = line.split("=")
        if len(parts) >= 2:
            package = parts[0]
            version = parts[1]
            requirements.append(f'"{package}=={version}"')

    print("install_requires = [")
    for req in requirements:
        print(f"    {req},")
    print("]")

conda_to_install_requires(r"C:\Users\Lak Lab\anaconda3\envs\patrick")  # Replace with your env name
