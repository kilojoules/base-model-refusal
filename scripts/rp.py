#!/usr/bin/env python3
"""Minimal RunPod GraphQL client.

Uses curl as the transport: RunPod sits behind Cloudflare, which rejects urllib's
default user agent with a 403.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, tempfile

API_KEY = (pathlib.Path(os.path.expanduser("~/.super_lab_run.pod"))
           .read_text().strip())
URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"


def gq(query, variables=None):
    body = {"query": query}
    if variables:
        body["variables"] = variables
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(body, fh)
        path = fh.name
    try:
        out = subprocess.run(
            ["curl", "-sS", "-X", "POST", URL,
             "-H", "Content-Type: application/json",
             "--data-binary", f"@{path}"],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            raise RuntimeError(f"curl failed: {out.stderr[:300]}")
        data = json.loads(out.stdout)
    finally:
        os.unlink(path)
    if "errors" in data:
        raise RuntimeError("GraphQL: " + json.dumps(data["errors"])[:600])
    return data["data"]


def add_pubkey(pub_path="~/.ssh/id_ed25519.pub"):
    new = pathlib.Path(os.path.expanduser(pub_path)).read_text().strip()
    cur = gq("{ myself { pubKey } }")["myself"]["pubKey"] or ""
    keys = [k.strip() for k in cur.split("\n") if k.strip()]
    newb = new.split()[1]
    if any(len(k.split()) > 1 and k.split()[1] == newb for k in keys):
        return {"status": "already_present", "count": len(keys)}
    combined = "\n".join(keys + [new])
    res = gq(
        "mutation($i:UpdateUserSettingsInput!){ updateUserSettings(input:$i){ id pubKey } }",
        {"i": {"pubKey": combined}},
    )
    got = [k for k in res["updateUserSettings"]["pubKey"].split("\n") if k.strip()]
    return {"status": "added", "count": len(got)}


def pods():
    q = """{ myself { pods { id name desiredStatus costPerHr
             machine { gpuDisplayName } runtime { uptimeInSeconds
             ports { ip isIpPublic privatePort publicPort type } } } } }"""
    return gq(q)["myself"]["pods"]


def deploy(name, gpu_type, gpu_count=2, volume_gb=600, disk_gb=60,
           image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
           vcpu=16, mem_gb=100, cloud="ALL"):
    q = """mutation($i: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $i) {
        id name imageName machineId costPerHr desiredStatus
      } }"""
    i = {
        "cloudType": cloud, "gpuCount": gpu_count, "gpuTypeId": gpu_type,
        "name": name, "imageName": image,
        "volumeInGb": volume_gb, "containerDiskInGb": disk_gb,
        "volumeMountPath": "/workspace",
        "minVcpuCount": vcpu, "minMemoryInGb": mem_gb,
        "ports": "22/tcp,8000/http",
        "startSsh": True,
        "dockerArgs": "",
        "env": [{"key": "HF_HOME", "value": "/workspace/hf"}],
    }
    return gq(q, {"i": i})["podFindAndDeployOnDemand"]


def terminate(pod_id):
    return gq("mutation($i:PodTerminateInput!){ podTerminate(input:$i) }",
              {"i": {"podId": pod_id}})


def stop(pod_id):
    return gq("mutation($i:PodStopInput!){ podStop(input:$i){ id desiredStatus } }",
              {"i": {"podId": pod_id}})


def ssh_target(pod_id):
    for p in pods():
        if p["id"] != pod_id:
            continue
        rt = p.get("runtime") or {}
        for prt in (rt.get("ports") or []):
            if prt["privatePort"] == 22 and prt["isIpPublic"]:
                return prt["ip"], prt["publicPort"]
    return None


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pods"
    if cmd == "pods":
        print(json.dumps(pods(), indent=2))
    elif cmd == "addkey":
        print(json.dumps(add_pubkey(), indent=2))
    elif cmd == "balance":
        print(json.dumps(gq("{ myself { clientBalance } }"), indent=2))
    elif cmd == "ssh":
        print(ssh_target(sys.argv[2]))
    elif cmd == "terminate":
        print(json.dumps(terminate(sys.argv[2]), indent=2))
    else:
        print(f"unknown command {cmd!r}")
